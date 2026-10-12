"""Pytest configuration for the JAX<->PyTorch equivalence suite.

Adds two directories to ``sys.path`` so both implementations are importable:

* ``_PY_DIR`` (``py/``) exposes the *PyTorch* implementation as ``common`` and the
  shared ``configs`` package;
* ``_TESTS_DIR`` (``py/tests/``) exposes the reference *JAX* implementation as
  ``jax_impl.common`` (kept under ``tests/`` solely for these equivalence tests).

Also forces CPU so the JAX/PyTorch comparisons are deterministic in float32.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PY_DIR = os.path.abspath(os.path.join(_TESTS_DIR, ".."))
for p in (_PY_DIR, _TESTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
