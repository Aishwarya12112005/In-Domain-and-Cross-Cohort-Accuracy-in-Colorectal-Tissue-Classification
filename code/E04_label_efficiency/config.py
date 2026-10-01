"""E04 configuration."""
SLICE_NAME = "E04 — Label-efficiency curves: how many annotated tiles do you need?"
QUESTION = (
    "Training on Kather-2016 and testing BOTH in-domain and on CRC-VAL-HE-7K, "
    "how does tissue-classification accuracy grow with the number of labelled "
    "tiles per class — and at what point does adding annotations stop paying? "
    "The curve is measured with repeated draws so the plateau is not a fluke of "
    "one subsample."
)

SEED = 0
TILE_SIZE = 112
MAX_PER_CLASS_TRAIN = 300
MAX_PER_CLASS_TEST = 300
LABEL_BUDGETS = (5, 10, 25, 50, 100, 200, 300)   # labelled tiles per class
N_REPEATS = 5
TEST_SIZE = 0.30
LOGREG = dict(l2=1e-3, lr=0.5, max_iter=350, class_weight="balanced")
