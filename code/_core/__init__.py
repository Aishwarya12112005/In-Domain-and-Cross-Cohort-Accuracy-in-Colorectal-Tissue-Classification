"""Shared numerical primitives for the CRC enhancement slices.

Only *primitives* live here (metrics, estimators, descriptors, IO, reporting).
The experiment logic itself is duplicated deliberately inside each
E##_* slice so that every slice is an independently runnable vertical
slice: ingest -> represent -> model -> evaluate -> report.
"""
__version__ = "1.0.0"
