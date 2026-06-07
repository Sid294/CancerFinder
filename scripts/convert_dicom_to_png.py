"""Convert DICOM files under a directory to PNG images.

Usage:
    python scripts/convert_dicom_to_png.py --src datasets/raw/dicom --dst datasets/images/png

If no args given, converts any .dcm files under `datasets/raw` into `datasets/images`.
"""
import argparse
from pathlib import Path
import pydicom
import numpy as np
from PIL import Image


def dicom_to_png(src_path: Path, dst_path: Path):
    try:
        ds = pydicom.dcmread(str(src_path))
        arr = ds.pixel_array
        # handle multi-frame by taking first frame
        if arr.ndim == 3:
            arr = arr[0]
        # normalize
        arr = arr.astype(float)
        arr = arr - arr.min()
        if arr.max() > 0:
            arr = arr / arr.max()
        arr = (arr * 255).astype('uint8')
        img = Image.fromarray(arr)
        img = img.convert('L')
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(dst_path)
        return True
    except Exception as e:
        print(f"Failed to convert {src_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='datasets/raw', help='Source root to search for .dcm files')
    parser.add_argument('--dst', type=str, default='datasets/images', help='Destination folder for PNGs')
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    files = list(src.rglob('*.dcm'))
    if not files:
        print(f"No .dcm files found under {src}")
        return
    print(f"Found {len(files)} DICOM files; converting to {dst}")
    count = 0
    for f in files:
        rel = f.relative_to(src)
        out = dst / rel.with_suffix('.png')
        if dicom_to_png(f, out):
            count += 1
    print(f"Converted {count}/{len(files)} files")

if __name__ == '__main__':
    main()
