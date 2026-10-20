#!/usr/bin/env python3
"""Verify the curated capture archive and every research-figure dependency."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(path: Path, expected: str) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"SHA-256 mismatch: {path.relative_to(ROOT)}")


def main() -> None:
    archive = ROOT / "results/quality-2026-10-20"
    entries = (archive / "SHA256SUMS").read_text().splitlines()
    for line in entries:
        expected, relative = line.split("  ", 1)
        verify(archive / relative, expected)
    manifest = json.loads((ROOT / "results/research-2026-10-20/figure-manifest.json").read_text())
    verify(ROOT / manifest["generator"], manifest["generator_sha256"])
    for section in ("input_sha256", "output_sha256"):
        for relative, expected in manifest[section].items():
            verify(ROOT / relative, expected)
    print(f"Verified {len(entries)} archive files, {len(manifest['input_sha256'])} figure inputs, "
          f"{len(manifest['output_sha256'])} figure outputs and the generator.")


if __name__ == "__main__":
    main()
