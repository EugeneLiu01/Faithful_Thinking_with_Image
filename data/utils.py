"""Resolve portable dataset paths without importing model or API dependencies."""

from pathlib import Path


def resolve_image_paths(example, data_dir):
    """Resolve image paths relative to the directory containing the Parquet."""
    images = example.get("image")
    if not images:
        return example

    def resolve(image_path):
        path = Path(image_path)
        return str(path if path.is_absolute() else (Path(data_dir) / path).resolve())

    resolved = [resolve(path) for path in images] if isinstance(images, list) else resolve(images)
    return {**example, "image": resolved}
