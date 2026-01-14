"""Deterministic fixed-point L2 certificate helpers.

This module is *not* a full zero-knowledge proof; it mirrors the fixed-point
encoding pipeline used by heavyweight RoFL/SDFL-style ZKPs, but keeps the math in
Python so that we can validate Δw bounds cheaply and consistently.

It exposes two helper functions:

- build_certificate(delta_params, scale, clip, tau)
- verify_certificate(delta_params, certificate, *, max_tau=None, max_scale=None)

Both functions accept the federated delta as a list of NumPy arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Tuple, List, Dict, Any, Optional

import numpy as np


@dataclass
class FixedPointConfig:
    scale: float = 1e4
    clip: Optional[float] = None
    tau: float = 10.0  # L2 bound in *float* space


def _as_float64(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.float64:
        return array
    return array.astype(np.float64, copy=False)


def _hash_and_l2(
    delta_params: Iterable[np.ndarray],
    scale: float,
    clip: Optional[float],
) -> Tuple[str, int, int]:
    """Return (sha256 hash, l2_sum_sq, num_elements) in fixed-point space."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    hasher = hashlib.sha256()
    total_l2 = 0
    total_elems = 0
    for tensor in delta_params:
        arr = _as_float64(np.asarray(tensor))
        if clip is not None and clip > 0:
            arr = np.clip(arr, -clip, clip)
        fixed = np.rint(arr * scale).astype(np.int64, copy=False)
        hasher.update(fixed.tobytes(order="C"))
        squares = fixed.astype(np.int64)
        total_l2 += int(np.sum(squares * squares))
        total_elems += fixed.size
    return hasher.hexdigest(), total_l2, total_elems


def build_certificate(
    delta_params: List[np.ndarray],
    *,
    scale: float = 1e4,
    clip: Optional[float] = None,
    tau: float = 10.0,
) -> Dict[str, Any]:
    """Build a deterministic certificate summarizing the L2 norm."""
    vector_hash, l2_sq, num_elems = _hash_and_l2(delta_params, scale, clip)
    tau_scaled = tau * scale
    tau_sq = int(np.round(tau_scaled * tau_scaled))
    if l2_sq > tau_sq:
        print(
            f"[FixedL2] l2_sq={l2_sq}, tau_sq={tau_sq}, "
            f"scale={scale}, clip={clip}, num_elems={num_elems}"
        )
        raise ValueError(
            f"L2^2={l2_sq} exceeds bound tau^2={tau_sq} (tau={tau})"
        )
    return {
        "scheme": "fixed_l2",
        "scale": scale,
        "clip": clip,
        "tau": tau,
        "tau_sq": tau_sq,
        "l2_sq": l2_sq,
        "num_elems": num_elems,
        "vector_hash": vector_hash,
    }


def verify_certificate(
    delta_params: List[np.ndarray],
    certificate: Dict[str, Any],
    *,
    max_tau: Optional[float] = None,
    max_scale: Optional[float] = None,
) -> bool:
    """Recompute deterministic certificate and compare it against payload."""
    if certificate.get("scheme") != "fixed_l2":
        return False
    clip = certificate.get("clip")
    scale = float(certificate.get("scale", 0.0))
    tau = float(certificate.get("tau", 0.0))
    if scale <= 0 or tau <= 0:
        return False
    if max_scale is not None and scale > max_scale:
        return False
    if max_tau is not None and tau > max_tau:
        return False
    expected_hash = certificate.get("vector_hash")
    expected_l2 = int(certificate.get("l2_sq", -1))
    tau_sq = int(certificate.get("tau_sq", -1))
    vector_hash, l2_sq, num_elems = _hash_and_l2(delta_params, scale, clip)
    if expected_hash and expected_hash != vector_hash:
        return False
    if expected_l2 != l2_sq:
        return False
    if tau_sq <= 0 or l2_sq > tau_sq:
        return False
    if certificate.get("num_elems") not in (None, num_elems):
        return False
    return True
