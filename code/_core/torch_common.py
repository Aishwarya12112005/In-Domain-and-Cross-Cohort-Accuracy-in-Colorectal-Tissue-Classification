"""
Shared PyTorch scaffolding for the `torch_impl/` scripts in each slice.

None of this runs on the machine that produced the classical results — that
host has no PyTorch and no GPU. These modules are the deep-learning counterpart
of each slice, written to run unchanged on Google Colab (free T4) or any CUDA
box, against the same real datasets in the same `datasets/` folder.

Install once:
    pip install torch torchvision timm

Every script keeps the *experimental design* of its classical counterpart —
same splits, same seeds, same metrics, same honest baselines — so the two sets
of numbers are directly comparable and the only thing that changes is the
feature extractor.
"""
from __future__ import annotations

import io
import os
import zipfile

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except Exception:                                    # pragma: no cover
    TORCH_AVAILABLE = False
    torch = None
    nn = object
    Dataset = object

import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class ZipTileTorchDataset(Dataset):
    """
    Reads tiles straight out of the dataset ZIP, mirroring `_core.zipsource`.

    The ZipFile handle is opened lazily per worker process, which is what makes
    this safe with DataLoader(num_workers > 0).
    """

    def __init__(self, zip_path, members, labels, size=224, train=False,
                 mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        self.zip_path = zip_path
        self.members = list(members)
        self.labels = np.asarray(labels)
        self.size = size
        self.train = train
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self._zf = None

    def _handle(self):
        if self._zf is None:
            self._zf = zipfile.ZipFile(self.zip_path)
        return self._zf

    def __len__(self):
        return len(self.members)

    def __getitem__(self, i):
        import cv2
        raw = self._handle().read(self.members[i])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k)
            img = _stain_jitter_rgb(img)
        x = (img.astype(np.float32) / 255.0 - self.mean) / self.std
        x = np.ascontiguousarray(x.transpose(2, 0, 1))
        return torch.from_numpy(x), int(self.labels[i])


def _stain_jitter_rgb(rgb, sigma=0.06):
    od = -np.log10(np.clip(rgb.astype(np.float32), 1, 255) / 255.0)
    a = 1.0 + np.random.normal(0, sigma, 3).astype(np.float32)
    b = np.random.normal(0, sigma, 3).astype(np.float32)
    od = od * a + b
    return np.clip(255.0 * (10 ** (-np.clip(od, 0, 4))), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
def build_backbone(name="resnet18", n_classes=7, pretrained=True):
    """
    Returns (model, feature_dim).  `timm` is used when available so that
    pathology foundation models (UNI, CONCH, Virchow) can be dropped in by name
    without changing any calling code.
    """
    try:
        import timm
        m = timm.create_model(name, pretrained=pretrained, num_classes=n_classes)
        feat_dim = getattr(m, "num_features", None)
        return m, feat_dim
    except Exception:
        import torchvision.models as tvm
        fn = getattr(tvm, name)
        m = fn(weights="DEFAULT" if pretrained else None)
        if hasattr(m, "fc"):
            feat_dim = m.fc.in_features
            m.fc = nn.Linear(feat_dim, n_classes)
        elif hasattr(m, "classifier"):
            feat_dim = (m.classifier[-1].in_features
                        if isinstance(m.classifier, nn.Sequential)
                        else m.classifier.in_features)
            if isinstance(m.classifier, nn.Sequential):
                m.classifier[-1] = nn.Linear(feat_dim, n_classes)
            else:
                m.classifier = nn.Linear(feat_dim, n_classes)
        else:
            raise RuntimeError(f"unsupported architecture {name}")
        return m, feat_dim


class SmallUNet(nn.Module):
    """Compact U-Net for the segmentation slices (E03, E07)."""

    def __init__(self, in_ch=3, out_ch=1, width=32):
        super().__init__()
        w = width

        def block(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

        self.e1, self.e2, self.e3, self.e4 = block(in_ch, w), block(w, 2*w), \
            block(2*w, 4*w), block(4*w, 8*w)
        self.pool = nn.MaxPool2d(2)
        self.u3 = nn.ConvTranspose2d(8*w, 4*w, 2, 2)
        self.d3 = block(8*w, 4*w)
        self.u2 = nn.ConvTranspose2d(4*w, 2*w, 2, 2)
        self.d2 = block(4*w, 2*w)
        self.u1 = nn.ConvTranspose2d(2*w, w, 2, 2)
        self.d1 = block(2*w, w)
        self.out = nn.Conv2d(w, out_ch, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


class MultiTaskNet(nn.Module):
    """Shared encoder, two heads: segmentation mask + differentiation grade (E03)."""

    def __init__(self, n_classes=6, width=32):
        super().__init__()
        self.unet = SmallUNet(3, 1, width)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(8 * width, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        u = self.unet
        e1 = u.e1(x); e2 = u.e2(u.pool(e1)); e3 = u.e3(u.pool(e2)); e4 = u.e4(u.pool(e3))
        d3 = u.d3(torch.cat([u.u3(e4), e3], 1))
        d2 = u.d2(torch.cat([u.u2(d3), e2], 1))
        d1 = u.d1(torch.cat([u.u1(d2), e1], 1))
        return u.out(d1), self.head(e4)


# ---------------------------------------------------------------------------
# Losses / training helpers
# ---------------------------------------------------------------------------
def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum(dim=(1, 2, 3)) + eps
    den = p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return 1 - (num / den).mean()


def set_seed(seed=0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def device():
    return torch.device("cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")


def train_classifier(model, train_loader, val_loader, epochs=10, lr=3e-4,
                     class_weights=None, log_every=1):
    """Standard AdamW + cosine schedule loop. Returns the per-epoch history."""
    dev = device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    w = None if class_weights is None else torch.tensor(
        class_weights, dtype=torch.float32, device=dev)
    crit = nn.CrossEntropyLoss(weight=w)
    hist = []
    for ep in range(epochs):
        model.train()
        tot = n = 0
        for x, y in train_loader:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            tot += float(loss) * x.size(0); n += x.size(0)
        sched.step()
        acc = evaluate_accuracy(model, val_loader) if val_loader else float("nan")
        hist.append({"epoch": ep, "train_loss": tot / max(n, 1), "val_accuracy": acc})
        if log_every and ep % log_every == 0:
            print(f"  epoch {ep:3d}  loss {tot / max(n, 1):.4f}  val_acc {acc:.4f}")
    return hist


@torch.no_grad()
def evaluate_accuracy(model, loader):
    dev = device()
    model.eval()
    correct = total = 0
    for x, y in loader:
        p = model(x.to(dev)).argmax(1).cpu()
        correct += int((p == y).sum()); total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def predict_proba(model, loader):
    dev = device()
    model.eval()
    P, Y = [], []
    for x, y in loader:
        P.append(Fn.softmax(model(x.to(dev)), dim=1).cpu().numpy())
        Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


@torch.no_grad()
def gradcam_free_occlusion(model, x, y_true, grid=8, fill=0.0):
    """Occlusion saliency for a CNN — identical protocol to the classical E02."""
    dev = device()
    model.eval()
    x = x.to(dev)
    base = Fn.softmax(model(x[None]), dim=1)[0, y_true].item()
    _, h, w = x.shape
    ch, cw = h // grid, w // grid
    sal = torch.zeros(grid, grid)
    for gy in range(grid):
        for gx in range(grid):
            occ = x.clone()
            occ[:, gy*ch:(gy+1)*ch, gx*cw:(gx+1)*cw] = fill
            p = Fn.softmax(model(occ[None]), dim=1)[0, y_true].item()
            sal[gy, gx] = base - p
    sal = torch.clamp(sal, min=0)
    sal = Fn.interpolate(sal[None, None], size=(h, w), mode="bicubic",
                         align_corners=False)[0, 0]
    return (sal / sal.max().clamp(min=1e-12)).cpu().numpy()


def gradcam(model, x, target_layer, y_true):
    """True Grad-CAM, the gradient-based counterpart to occlusion saliency."""
    dev = device()
    model.eval()
    feats, grads = {}, {}

    def fh(_m, _i, o):
        feats["v"] = o.detach()

    def bh(_m, _gi, go):
        grads["v"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fh)
    h2 = target_layer.register_full_backward_hook(bh)
    x = x[None].to(dev).requires_grad_(True)
    out = model(x)
    model.zero_grad(set_to_none=True)
    out[0, y_true].backward()
    h1.remove(); h2.remove()
    a = feats["v"][0]
    g = grads["v"][0].mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((a * g).sum(0))
    cam = Fn.interpolate(cam[None, None], size=x.shape[-2:], mode="bicubic",
                         align_corners=False)[0, 0]
    return (cam / cam.max().clamp(min=1e-12)).detach().cpu().numpy()
