"""Content and perceptual hashes used before real-data splitting."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image, hash_size: int = 8) -> str:
    """Return a 64-bit dHash by comparing adjacent grayscale pixels."""

    if hash_size <= 0:
        raise ValueError("hash_size must be positive")
    grayscale = image.convert("L").resize((hash_size + 1, hash_size))
    pixels = list(grayscale.getdata())
    value = 0
    for row in range(hash_size):
        offset = row * (hash_size + 1)
        for column in range(hash_size):
            value = (value << 1) | (
                pixels[offset + column] > pixels[offset + column + 1]
            )
    return f"{value:0{hash_size * hash_size // 4}x}"


def hamming_distance(first: str, second: str) -> int:
    if len(first) != len(second):
        raise ValueError("perceptual hashes must have equal length")
    return (int(first, 16) ^ int(second, 16)).bit_count()

