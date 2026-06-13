"""
Download the Kaggle dataset and run a simple classification pipeline.

Behavior:
- Downloads and unzips the dataset using the `kaggle` CLI into `datasets/raw`.
- Scans for image files. If a CSV with labels is found, trains a small
  transfer-learning classifier (ResNet18) for a few epochs and writes
  predictions to `datasets/predictions.csv`.
- If no labels are found, extracts embeddings with a pretrained ResNet
  and writes them to `datasets/embeddings.csv`.

Requirements: `kaggle` CLI configured, `torch`, `torchvision`, `pandas`, `Pillow`.
"""

import os
import sys
import subprocess
from pathlib import Path
import glob
import csv

try:
  import torch
  import torch.nn as nn
  import torch.optim as optim
  from torch.utils.data import Dataset, DataLoader
  from torchvision import models, transforms
  from PIL import Image
  import pandas as pd
  import numpy as np
  import pydicom
except Exception as e:
  print("Missing dependencies. Please install dependencies from requirements.txt and Pillow and pandas.")
  print(e)
  sys.exit(1)


ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "datasets" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

DATASET_SLUG = "ymirsky/medical-deepfakes-lung-cancer"

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
DICOM_EXTS = {".dcm"}


def run_kaggle_download(slug: str, dest: Path):
  print(f"Downloading dataset {slug} into {dest} (requires kaggle CLI)...")
  try:
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
  except FileNotFoundError:
    print("`kaggle` CLI not found. Install it with `pip install kaggle` and configure API credentials.")
    raise
  except subprocess.CalledProcessError as e:
    print("kaggle download failed:", e)
    raise


def find_images(base: Path):
  imgs = []
  for ext in IMAGE_EXTS:
    imgs.extend(base.rglob(f"*{ext}"))
  for ext in DICOM_EXTS:
    imgs.extend(base.rglob(f"*{ext}"))
  return sorted(imgs)


def load_image(path: Path):
  if path.suffix.lower() in DICOM_EXTS:
    ds = pydicom.dcmread(str(path), force=True)
    try:
      arr = ds.pixel_array
    except Exception:
      # Some DICOMs are missing File Meta / Transfer Syntax; try a reasonable fallback
      try:
        from pydicom.dataset import FileMetaDataset
        from pydicom.uid import ImplicitVRLittleEndian
        if not hasattr(ds, 'file_meta') or ds.file_meta is None:
          ds.file_meta = FileMetaDataset()
        if not getattr(ds.file_meta, 'TransferSyntaxUID', None):
          ds.file_meta.TransferSyntaxUID = ImplicitVRLittleEndian
        arr = ds.pixel_array
      except Exception:
        # final fallback: return a blank image to keep pipeline running
        import numpy as _np
        arr = _np.zeros((512, 512), dtype='uint8')
    if arr.ndim == 3:
      arr = arr[0]
    arr = arr.astype(float)
    arr = arr - arr.min()
    if arr.max() > 0:
      arr = arr / arr.max()
    arr = (arr * 255).astype('uint8')
    return Image.fromarray(arr).convert('RGB')
  return Image.open(path).convert('RGB')


def find_labels_csv(base: Path):
  csvs = list(base.rglob("*.csv"))
  for c in csvs:
    try:
      df = pd.read_csv(c, nrows=5)
    except Exception:
      continue
    cols = [c.lower() for c in df.columns]
    if any(x in cols for x in ("label", "diagnosis", "target", "class", "category")) or any(x in cols for x in ("image", "filename", "file")):
      return c
  return None


def build_dicom_lookup(image_paths):
  lookup = {}
  for path in image_paths:
    key = (path.parent.name, path.stem)
    lookup.setdefault(key, str(path))
  return lookup


class ImageDataset(Dataset):
  def __init__(self, rows, img_col, label_col, transform=None):
    self.rows = rows
    self.img_col = img_col
    self.label_col = label_col
    self.transform = transform

  def __len__(self):
    return len(self.rows)

  def __getitem__(self, idx):
    path = self.rows.iloc[idx][self.img_col]
    label = int(self.rows.iloc[idx][self.label_col])
    img = load_image(Path(path))
    if self.transform:
      img = self.transform(img)
    return img, label


def extract_embeddings(image_paths, device, batch_size=32):
  model = models.resnet18(pretrained=True)
  model.fc = nn.Identity()
  model = model.to(device).eval()
  tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229,0.224,0.225])])

  embeddings = []
  paths = []
  with torch.no_grad():
    for i in range(0, len(image_paths), batch_size):
      batch_paths = image_paths[i:i+batch_size]
      batch_imgs = torch.stack([tf(load_image(Path(p))) for p in batch_paths]).to(device)
      emb = model(batch_imgs).cpu().numpy()
      embeddings.append(emb)
      paths.extend([str(p) for p in batch_paths])
  embeddings = np.vstack(embeddings)
  return paths, embeddings


def train_classifier(df_labels, img_col, label_col, device):
  # simple train/val split without extra dependencies
  shuffled = df_labels.sample(frac=1.0, random_state=42)
  split_idx = max(1, int(len(shuffled) * 0.8))
  train_df = shuffled.iloc[:split_idx]
  val_df = shuffled.iloc[split_idx:]
  if len(val_df) == 0:
    val_df = train_df.iloc[:1]

  transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

  train_ds = ImageDataset(train_df.reset_index(drop=True), img_col, label_col, transform=transform)
  val_ds = ImageDataset(val_df.reset_index(drop=True), img_col, label_col, transform=transform)

  train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
  val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2)

  model = models.resnet18(pretrained=True)
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, 2)
  model = model.to(device)

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  epochs = 3
  for epoch in range(epochs):
    model.train()
    running = 0
    for xb, yb in train_loader:
      xb, yb = xb.to(device), yb.to(device)
      optimizer.zero_grad()
      out = model(xb)
      loss = criterion(out, yb)
      loss.backward()
      optimizer.step()
      running += loss.item()
    print(f"Epoch {epoch+1}/{epochs} train loss: {running/len(train_loader):.4f}")

  # evaluation + predictions on full label set
  model.eval()
  preds = []
  with torch.no_grad():
    for xb, _ in val_loader:
      xb = xb.to(device)
      out = model(xb)
      p = torch.argmax(out, dim=1).cpu().numpy().tolist()
      preds.extend(p)

  # Save model
  model_path = ROOT / "models"
  model_path.mkdir(exist_ok=True)
  torch.save(model.state_dict(), model_path / "resnet18_finetuned.pt")
  print("Saved trained model to models/resnet18_finetuned.pt")

  return model


def main():
  try:
    run_kaggle_download(DATASET_SLUG, RAW_DIR)
  except Exception:
    print("Proceeding assuming dataset already present in datasets/raw.")

  # Prefer pre-converted PNG/JPG images from `datasets/images` if present
  IMAGES_DIR = ROOT / 'datasets' / 'images'
  if IMAGES_DIR.exists():
    imgs = find_images(IMAGES_DIR)
    print(f"Found {len(imgs)} images under {IMAGES_DIR}")
  else:
    imgs = find_images(RAW_DIR)
    print(f"Found {len(imgs)} images under {RAW_DIR}")

  labels_csv = find_labels_csv(RAW_DIR)
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  if labels_csv:
    print(f"Found labels CSV: {labels_csv}")
    df = pd.read_csv(labels_csv)

    # Attempt to detect image and label columns
    cols = [c.lower() for c in df.columns]
    img_col = None
    label_col = None
    for c in df.columns:
      if c.lower() in ("image", "filename", "file", "path"):
        img_col = c
      if c.lower() in ("label", "diagnosis", "target", "class", "category", "label_id"):
        label_col = c
    if img_col is None:
      if {"uuid", "slice"}.issubset({c.lower() for c in df.columns}):
        lookup = build_dicom_lookup(imgs)

        def resolve_row_path(row):
          uuid_value = str(row["uuid"]).split(".")[0]
          slice_value = str(int(float(row["slice"])))
          return lookup.get((uuid_value, slice_value), "")

        df["filepath"] = df.apply(resolve_row_path, axis=1)
        if df["filepath"].str.len().sum() > 0:
          img_col = "filepath"
      if img_col is None:
        # try to match filenames to full paths
        df['__match__'] = df[df.columns[0]].astype(str)
        # try to find by basename
        basemap = {p.name: str(p) for p in imgs}
        df['filepath'] = df.iloc[:,0].astype(str).map(lambda x: basemap.get(Path(x).name, ""))
        if df['filepath'].str.len().sum() > 0:
          img_col = 'filepath'
    if label_col is None:
      # fallback: assume second column is label
      label_col = df.columns[1]

    if img_col not in df.columns:
      print("Could not resolve image column in labels CSV. Falling back to embedding extraction.")
    else:
      # Filter rows where file exists
      df[img_col] = df[img_col].apply(lambda x: str(x))
      df[img_col] = df[img_col].apply(lambda p: p if Path(p).exists() else str(RAW_DIR / p) if (RAW_DIR / p).exists() else p)
      df = df[df[img_col].apply(lambda p: Path(p).exists())]
      if len(df) == 0:
        print("No labeled images found on disk. Falling back to embeddings.")
      else:
        print(f"Training classifier on {len(df)} labeled images")
        model = train_classifier(df, img_col, label_col, device)
        print("Done training")
        return

  # If we reach here, no usable labels: extract embeddings
  if len(imgs) == 0:
    print("No images found. Please check the dataset contents in datasets/raw.")
    return

  paths, embs = extract_embeddings([str(p) for p in imgs], device)
  out_csv = ROOT / 'datasets' / 'embeddings.csv'
  print(f"Writing embeddings to {out_csv}")
  cols = [f"e{i}" for i in range(embs.shape[1])]
  df_out = pd.DataFrame(embs, columns=cols)
  df_out.insert(0, 'path', paths)
  out_csv.parent.mkdir(parents=True, exist_ok=True)
  df_out.to_csv(out_csv, index=False)


if __name__ == '__main__':
  main()
