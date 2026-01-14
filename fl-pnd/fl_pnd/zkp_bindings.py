"""Utility bindings to call RoFL's librofl_crypto for L2 range proofs."""
from __future__ import annotations

import base64
import ctypes
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np


class ZKPBindingError(RuntimeError):
    """Raised when the native RoFL crypto library returns an error."""


class _PyVec(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("len", ctypes.c_size_t),
    ]


class _PyRes(ctypes.Structure):
    _fields_ = [
        ("ret", ctypes.c_size_t),
        ("msg", ctypes.c_char_p),
        ("res", ctypes.c_void_p),
    ]


@dataclass
class L2ProofArtifacts:
    """Container for the serialized proof blobs returned by librofl_crypto."""

    commitments: bytes
    rand_proof: bytes
    range_proof: bytes
    square_commit: bytes
    prove_range_bits: int
    n_partition: int
    vector_len: int
    l2_norm: float

    def to_payload(self) -> dict:
        def _enc(blob: bytes) -> str:
            return base64.b64encode(blob).decode("ascii")

        return {
            "prove_range_bits": self.prove_range_bits,
            "n_partition": self.n_partition,
            "vector_len": self.vector_len,
            "l2_norm": self.l2_norm,
            "commitments_b64": _enc(self.commitments),
            "rand_proof_b64": _enc(self.rand_proof),
            "range_proof_b64": _enc(self.range_proof),
            "square_commit_b64": _enc(self.square_commit),
        }

    @classmethod
    def from_payload(cls, payload: dict) -> "L2ProofArtifacts":
        def _dec(value: str) -> bytes:
            return base64.b64decode(value.encode("ascii"))

        return cls(
            commitments=_dec(payload["commitments_b64"]),
            rand_proof=_dec(payload["rand_proof_b64"]),
            range_proof=_dec(payload["range_proof_b64"]),
            square_commit=_dec(payload["square_commit_b64"]),
            prove_range_bits=int(payload["prove_range_bits"]),
            n_partition=int(payload.get("n_partition", 1)),
            vector_len=int(payload.get("vector_len", 0)),
            l2_norm=float(payload.get("l2_norm", 0.0)),
        )


class RoFLL2Engine:
    """Thin ctypes-based wrapper around librofl_crypto for L2 range proofs."""

    def __init__(self, lib_path: Optional[str] = None):
        if lib_path is None:
            lib_path = Path(__file__).resolve().parent.parent / "zkp" / "librofl_crypto.so"
        self.lib_path = Path(lib_path).resolve()
        if not self.lib_path.exists():
            raise FileNotFoundError(f"librofl_crypto.so not found at {self.lib_path}")
        self._lib = ctypes.cdll.LoadLibrary(str(self.lib_path))
        self._configure_signatures()

    def _configure_signatures(self) -> None:
        size_t = ctypes.c_size_t
        uchar_p = ctypes.POINTER(ctypes.c_ubyte)
        float_p = ctypes.POINTER(ctypes.c_float)

        self._lib.create_random_blinding_vector.argtypes = [size_t]
        self._lib.create_random_blinding_vector.restype = _PyVec

        self._lib.create_l2proof.argtypes = [
            float_p,
            size_t,
            uchar_p,
            size_t,
            uchar_p,
            size_t,
            size_t,
            size_t,
        ]
        self._lib.create_l2proof.restype = _PyRes

        self._lib.verify_l2proof.argtypes = [
            uchar_p,
            size_t,
            uchar_p,
            size_t,
            uchar_p,
            uchar_p,
            size_t,
        ]
        self._lib.verify_l2proof.restype = _PyRes

    @staticmethod
    def _pyvec_to_bytes(pyvec: _PyVec) -> bytes:
        if not pyvec.data or pyvec.len == 0:
            return b""
        return ctypes.string_at(pyvec.data, pyvec.len)

    @staticmethod
    def _pyvec_to_vec(pyvec: _PyVec):
        if not pyvec.data or pyvec.len == 0:
            return []
        array_type = _PyVec * pyvec.len
        data = ctypes.cast(pyvec.data, ctypes.POINTER(array_type)).contents
        return list(data)

    @staticmethod
    def _raise_on_error(pyres: _PyRes) -> ctypes.c_void_p:
        if pyres.ret != 0:
            msg = ctypes.string_at(pyres.msg).decode("utf-8") if pyres.msg else "Unknown error"
            raise ZKPBindingError(msg)
        return pyres.res

    @staticmethod
    def _pyres_to_bool(pyres: _PyRes) -> bool:
        ptr = RoFLL2Engine._raise_on_error(pyres)
        bool_ptr = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_bool))
        return bool(bool_ptr.contents.value)

    @staticmethod
    def _pyres_to_pyvec(pyres: _PyRes) -> _PyVec:
        ptr = RoFLL2Engine._raise_on_error(pyres)
        return ctypes.cast(ptr, ctypes.POINTER(_PyVec)).contents

    @staticmethod
    def _bytes_to_uchar_ptr(blob: bytes):
        if not blob:
            return None, 0, None
        buf = ctypes.create_string_buffer(blob)
        ptr = ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))
        return ptr, len(blob), buf

    def _random_blinding_bytes(self, length: int) -> bytes:
        pyvec = self._lib.create_random_blinding_vector(ctypes.c_size_t(length))
        return self._pyvec_to_bytes(pyvec)

    @staticmethod
    def flatten_delta(delta_params: Sequence[np.ndarray]) -> np.ndarray:
        if not delta_params:
            return np.array([], dtype=np.float32)
        flat = np.concatenate([np.asarray(arr, dtype=np.float32).ravel() for arr in delta_params])
        return flat.astype(np.float32, copy=False)

    def prove_l2_bound(
        self,
        delta_params: Sequence[np.ndarray],
        range_bits: int,
        n_partition: int,
    ) -> Optional[L2ProofArtifacts]:
        vector = self.flatten_delta(delta_params)
        vector_len = int(vector.size)
        if vector_len == 0:
            return None
        blinding_1 = self._random_blinding_bytes(vector_len)
        blinding_2 = self._random_blinding_bytes(vector_len)

        ptr_b1, len_b1, buf1 = self._bytes_to_uchar_ptr(blinding_1)
        ptr_b2, len_b2, buf2 = self._bytes_to_uchar_ptr(blinding_2)
        float_ptr = vector.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        pyres = self._lib.create_l2proof(
            float_ptr,
            ctypes.c_size_t(vector_len),
            ptr_b1,
            ctypes.c_size_t(len_b1),
            ptr_b2,
            ctypes.c_size_t(len_b2),
            ctypes.c_size_t(range_bits),
            ctypes.c_size_t(max(1, n_partition)),
        )
        del buf1, buf2
        pyvec = self._pyres_to_pyvec(pyres)
        components = self._pyvec_to_vec(pyvec)
        if len(components) != 4:
            raise ZKPBindingError(
                f"Unexpected create_l2proof payload length {len(components)} (expected 4)"
            )
        l2_norm = float(np.linalg.norm(vector.astype(np.float64)))
        return L2ProofArtifacts(
            commitments=self._pyvec_to_bytes(components[1]),
            rand_proof=self._pyvec_to_bytes(components[0]),
            range_proof=self._pyvec_to_bytes(components[2]),
            square_commit=self._pyvec_to_bytes(components[3]),
            prove_range_bits=int(range_bits),
            n_partition=max(1, int(n_partition)),
            vector_len=vector_len,
            l2_norm=l2_norm,
        )

    def verify_l2_proof(self, artifacts: L2ProofArtifacts) -> bool:
        ptr_commit, len_commit, buf_commit = self._bytes_to_uchar_ptr(artifacts.commitments)
        ptr_rand, len_rand, buf_rand = self._bytes_to_uchar_ptr(artifacts.rand_proof)
        ptr_range, len_range, buf_range = self._bytes_to_uchar_ptr(artifacts.range_proof)
        ptr_square, len_square, buf_square = self._bytes_to_uchar_ptr(artifacts.square_commit)

        pyres = self._lib.verify_l2proof(
            ptr_commit,
            ctypes.c_size_t(len_commit),
            ptr_rand,
            ctypes.c_size_t(len_rand),
            ptr_range,
            ptr_square,
            ctypes.c_size_t(artifacts.prove_range_bits),
        )
        del buf_commit, buf_rand, buf_range, buf_square
        return self._pyres_to_bool(pyres)


def compute_payload_hash(serialized_delta: bytes) -> str:
    return hashlib.sha256(serialized_delta).hexdigest()

