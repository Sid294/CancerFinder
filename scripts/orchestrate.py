"""Orchestrate end-to-end: parse XMLs, download DICOMs from TCIA, convert DICOMs to PNGs, and run `load_images.py`.

Usage:
    python scripts/orchestrate.py --api-key YOUR_TCIA_API_KEY

Notes: TCIA API may require different endpoints or rate limits. This script attempts a best-effort download.
"""
import subprocess
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd, check=True):
    print('>',' '.join(cmd))
    subprocess.run(cmd, check=check)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', type=str, default=None)
    args = parser.parse_args()

    # 1) parse XMLs
    run(['python','scripts/parse_lidc_xml.py'])

    # 2) download series (requires API key for many TCIA endpoints)
    dl_cmd = ['python','scripts/download_tcia_series.py','--csv','datasets/series_uids.csv']
    if args.api_key:
        dl_cmd += ['--api-key', args.api_key]
    run(dl_cmd)

    # 3) convert DICOMs to PNGs
    run(['python','scripts/convert_dicom_to_png.py','--src','datasets/raw/dicom','--dst','datasets/images'])

    # 4) run load_images pipeline (training/embeddings)
    run(['python','load_images.py'])

if __name__ == '__main__':
    main()
