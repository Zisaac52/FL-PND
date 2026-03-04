from __future__ import annotations

import base64
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

DEFAULT_BIN = (
    Path(__file__).resolve().parents[1]
    / "zkp-groth16-l2"
    / "target"
    / "release"
    / "zkp-groth16-l2"
)


class Groth16Error(RuntimeError):
    """Raised when Groth16 CLI commands fail."""


def _default_bin_path() -> Path:
    return DEFAULT_BIN


def _ensure_file(path: Path, kind: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Groth16 {kind} not found at {path}")
    return path


def _flatten_params(delta_params: Sequence[np.ndarray]) -> np.ndarray:
    if not delta_params:
        return np.array([], dtype=np.float32)
    flat = np.concatenate([np.asarray(arr, dtype=np.float32).ravel() for arr in delta_params])
    return flat.astype(np.float32, copy=False)


class Groth16Engine:
    def __init__(
        self,
        *,
        bin_path: str | Path | None = None,
        pk_path: str | Path | None = None,
        vk_path: str | Path | None = None,
    ) -> None:
        bin_candidate = Path(bin_path) if bin_path else _default_bin_path()
        self.bin_path = _ensure_file(bin_candidate, "binary")
        self.pk_path = Path(pk_path).resolve() if pk_path else None
        self.vk_path = Path(vk_path).resolve() if vk_path else None

    def prove_delta(
        self,
        delta_params: Sequence[np.ndarray],
        *,
        scale: float,
        tau: float,
        clip: float | None,
        diff_bits: int,
        delta_hash: str | None = None,
    ) -> dict:
        if self.pk_path is None:
            raise Groth16Error("proving key path not configured")
        flat = _flatten_params(delta_params)
        with tempfile.TemporaryDirectory(prefix="groth16_") as tmpdir:
            tmp = Path(tmpdir)
            delta_path = tmp / "delta.bin"
            flat.astype("<f4").tofile(delta_path)
            witness_path = tmp / "witness.bin"
            public_path = tmp / "public.bin"
            proof_path = tmp / "proof.bin"

            witness_cmd = [
                "witness",
                "--input",
                str(delta_path),
                "--len",
                str(flat.size),
                "--scale",
                str(scale),
                "--tau",
                str(tau),
                "--diff-bits",
                str(max(1, diff_bits)),
                "--witness-out",
                str(witness_path),
                "--public-out",
                str(public_path),
            ]
            if clip is not None:
                witness_cmd += ["--clip", str(clip)]

            witness_stdout = self._run_cli(witness_cmd, capture_output=True)
            stats = self._parse_witness_stats(witness_stdout)
            prove_cmd = [
                "prove",
                "--pk",
                str(self.pk_path),
                "--witness",
                str(witness_path),
                "--public",
                str(public_path),
                "--proof",
                str(proof_path),
            ]
            prove_time = self._run_timed(prove_cmd)
            proof_bytes = proof_path.read_bytes()
            public_bytes = public_path.read_bytes()

        payload = {
            "scheme": "groth16",
            "proof_b64": base64.b64encode(proof_bytes).decode("ascii"),
            "public_b64": base64.b64encode(public_bytes).decode("ascii"),
            "num_values": stats["samples"],
            "l2_sq": stats["l2_sq"],
            "tau_sq": stats["tau_sq"],
            "prove_time": prove_time,
            "proof_bytes": len(proof_bytes),
            "delta_hash": delta_hash,
            "scale": scale,
            "tau": tau,
            "clip": clip,
            "diff_bits": diff_bits,
        }
        return payload

    def verify(self, proof_b64: str, public_b64: str) -> float:
        if self.vk_path is None:
            raise Groth16Error("verifying key path not configured")
        proof_raw = base64.b64decode(proof_b64.encode("ascii"))
        public_raw = base64.b64decode(public_b64.encode("ascii"))
        with tempfile.TemporaryDirectory(prefix="groth16_") as tmpdir:
            tmp = Path(tmpdir)
            proof_path = tmp / "proof.bin"
            public_path = tmp / "public.bin"
            proof_path.write_bytes(proof_raw)
            public_path.write_bytes(public_raw)
            verify_cmd = [
                "verify",
                "--vk",
                str(self.vk_path),
                "--public",
                str(public_path),
                "--proof",
                str(proof_path),
            ]
            verify_time = self._run_timed(verify_cmd)
        return verify_time

    def _run_cli(self, args: Iterable[str], capture_output: bool = False) -> str:
        cmd = [str(self.bin_path)] + list(args)
        try:
            proc = subprocess.run(
                cmd,
                check=True,
                text=True,
                capture_output=capture_output,
            )
        except subprocess.CalledProcessError as exc:
            cmd_str = " ".join(cmd)
            raise Groth16Error(
                f"Groth16 command failed: {cmd_str}\n{exc.stderr}"
            ) from exc
        return proc.stdout if capture_output else ""

    def _run_timed(self, args: Iterable[str]) -> float:
        start = time.perf_counter()
        self._run_cli(args)
        return time.perf_counter() - start

    @staticmethod
    def _parse_witness_stats(stdout: str) -> dict:
        samples = None
        l2_sq = None
        tau_sq = None
        for line in stdout.splitlines()[::-1]:
            line = line.strip()
            if line.startswith("witness prepared"):
                parts = line.replace(",", " " ).split()
                for token in parts:
                    if token.startswith("samples="):
                        samples = int(token.split("=", 1)[1])
                    elif token.startswith("l2_sq="):
                        l2_sq = int(token.split("=", 1)[1])
                    elif token.startswith("tau_sq="):
                        tau_sq = int(token.split("=", 1)[1])
                break
        if samples is None or l2_sq is None or tau_sq is None:
            raise Groth16Error(f"Failed to parse witness stats from: {stdout}")
        return {"samples": samples, "l2_sq": l2_sq, "tau_sq": tau_sq}
