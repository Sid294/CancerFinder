"""Download TCIA series ZIPs given a CSV of SeriesInstanceUIDs (from parse_lidc_xml.py).

Writes ZIPs to datasets/raw/dicom_zips and extracts them to datasets/raw/dicom/<series_uid>/

Usage:
    python scripts/download_tcia_series.py --csv datasets/series_uids.csv --api-key YOUR_KEY

If no --api-key provided, will try with no auth header.
"""
import argparse
from pathlib import Path
import requests
import zipfile
import io


def download_series(uid, out_dir: Path, api_key=None):
    url = f"https://services.cancerimagingarchive.net/services/v4/TCIA/query/getImage?SeriesInstanceUID={uid}"
    headers = {}
    if api_key:
        headers['api_key'] = api_key
    try:
        r = requests.get(url, headers=headers, timeout=120)
        r.raise_for_status()
        # Some responses might be a redirect to a download URL; handle content-type
        if 'zip' in r.headers.get('Content-Type', '') or r.content.startswith(b'PK'):
            outzip = out_dir / f"{uid}.zip"
            outzip.parent.mkdir(parents=True, exist_ok=True)
            outzip.write_bytes(r.content)
            # extract
            extract_dir = out_dir.parent / 'dicom' / uid
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                z.extractall(path=extract_dir)
            return True, str(extract_dir)
        else:
            # Not zip — write raw
            outzip = out_dir / f"{uid}.bin"
            outzip.parent.mkdir(parents=True, exist_ok=True)
            outzip.write_bytes(r.content)
            return False, str(outzip)
    except Exception as e:
        return False, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='datasets/series_uids.csv')
    parser.add_argument('--api-key', type=str, default=None)
    args = parser.parse_args()

    csvp = Path(args.csv)
    if not csvp.exists():
        print(f"CSV not found: {csvp}")
        return
    out_dir = Path('datasets/raw/dicom_zips')
    out_dir.mkdir(parents=True, exist_ok=True)

    import csv
    with csvp.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            series_uid = row.get('series_uid')
            if not series_uid:
                continue
            series_uid = series_uid.split(',')[0]
            print(f"Downloading series {series_uid} ...")
            ok, info = download_series(series_uid, out_dir, api_key=args.api_key)
            if ok:
                print(f"Downloaded and extracted to {info}")
            else:
                print(f"Failed to download {series_uid}: {info}")

if __name__ == '__main__':
    main()
