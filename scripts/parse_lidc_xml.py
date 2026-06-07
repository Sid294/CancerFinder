"""Parse LIDC XML files to extract SeriesInstanceUIDs and other identifiers.

Writes a CSV `datasets/series_uids.csv` with columns: xml_file, series_uid, study_uid

Usage:
    python scripts/parse_lidc_xml.py
"""
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import csv

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'datasets' / 'raw'
OUT = ROOT / 'datasets' / 'series_uids.csv'

UID_RE = re.compile(r"[0-9]+(\.[0-9]+)+")

rows = []
for xml in RAW.rglob('*.xml'):
    try:
        tree = ET.parse(xml)
        root = tree.getroot()
        text = ET.tostring(root, encoding='utf-8', method='xml').decode('utf-8')
        # find all UID-like tokens
        uids = UID_RE.findall(text)
        # try to find SeriesInstanceUID tags
        series_uids = []
        study_uids = []
        for elem in root.iter():
            tag = elem.tag.lower()
            if 'seriesinstanceuid' in tag or 'series_instance_uid' in tag or 'seriesuid' in tag:
                val = (elem.text or '').strip()
                if val:
                    series_uids.append(val)
            if 'studyinstanceuid' in tag or 'study_instance_uid' in tag or 'studyuid' in tag:
                val = (elem.text or '').strip()
                if val:
                    study_uids.append(val)
        if not series_uids:
            # fallback: take any UID-like string
            series_uids = uids[:1]
        if not study_uids and len(uids) > 1:
            study_uids = [uids[1]]
        rows.append((str(xml.relative_to(ROOT)), ','.join(series_uids), ','.join(study_uids)))
    except Exception as e:
        print(f"Failed to parse {xml}: {e}")

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['xml_file','series_uid','study_uid'])
    for r in rows:
        writer.writerow(r)

print(f"Wrote {len(rows)} rows to {OUT}")
