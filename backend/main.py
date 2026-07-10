from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from scripts.load_images import find_images, find_labels_csv, load_image, build_dicom_lookup

# Use correct path to datasets/raw
RAW_DIR = Path(__file__).resolve().parent.parent / "datasets" / "raw"

ROOT = Path(__file__).resolve().parent
epochs = 25
lr = 0.0001  # Lower learning rate for fine-tuning
weight_decay = 1e-4  # L2 regularization
IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Label mapping: FB/TB = non-cancerous (0), FM/TM = cancerous (1)
LABEL_MAP = {"FB": 0, "TB": 0, "FM": 1, "TM": 1}


def build_model():
    try:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
    except Exception:
        model = models.resnet50(pretrained=True)
    # Replace fc with a simple linear layer (no wrapper)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def make_train_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # Increased from 2
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Add color augmentation
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_eval_transform():
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class ImageDataset(Dataset):
    def __init__(self, frame, path_col, label_col, transform=None):
        self.frame = frame.reset_index(drop=True)
        self.path_col = path_col
        self.label_col = label_col
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = Path(self.frame.loc[index, self.path_col])
        label = int(self.frame.loc[index, self.label_col])
        image = load_image(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def resolve_labeled_images():
    image_paths = find_images(RAW_DIR)
    labels_csv = find_labels_csv(RAW_DIR)

    if labels_csv is None:
        return None, None

    frame = pd.read_csv(labels_csv)
    columns = {column.lower(): column for column in frame.columns}
    
    # Check for 'type' column (FB, FM, TB, TM) and map to binary labels
    type_col = columns.get("type")
    if type_col is not None:
        # Map type codes to binary labels
        frame["label"] = frame[type_col].map(LABEL_MAP)
        label_col = "label"
    else:
        label_col = columns.get("label")
        if label_col is None:
            return None, None

    path_col = None
    if {"uuid", "slice"}.issubset(columns):
        lookup = build_dicom_lookup(image_paths)

        def resolve_row(row):
            uuid_value = str(row[columns["uuid"]]).split(".")[0]
            slice_value = str(int(float(row[columns["slice"]])))
            return lookup.get((uuid_value, slice_value), "")

        frame["filepath"] = frame.apply(resolve_row, axis=1)
        if frame["filepath"].str.len().sum() > 0:
            path_col = "filepath"

    if path_col is None:
        return None, None

    frame = frame[frame[path_col].astype(str).str.len() > 0].copy()
    frame[label_col] = frame[label_col].astype(int)
    return frame, (path_col, label_col)


def train():
    frame, cols = resolve_labeled_images()
    if frame is None:
        raise RuntimeError("Could not resolve labeled images from datasets/raw")

    path_col, label_col = cols
    shuffled = frame.sample(frac=1.0, random_state=42)
    split_idx = max(1, int(len(shuffled) * 0.8))
    train_frame = shuffled.iloc[:split_idx].copy()
    val_frame = shuffled.iloc[split_idx:].copy()
    if len(val_frame) == 0:
        val_frame = train_frame.iloc[:1].copy()

    train_transform = make_train_transform()
    val_transform = make_eval_transform()  # Use eval transform for validation

    train_loader = DataLoader(
        ImageDataset(train_frame, path_col, label_col, transform=train_transform),
        batch_size=16,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ImageDataset(val_frame, path_col, label_col, transform=val_transform),  # Fixed: use val_transform
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model().to(device)

    class_counts = train_frame[label_col].value_counts().to_dict()
    neg_count = float(class_counts.get(0, 1))
    pos_count = float(class_counts.get(1, 1))
    total_count = neg_count + pos_count
    class_weights = torch.tensor([
        total_count / max(1.0, 2.0 * neg_count),
        total_count / max(1.0, 2.0 * pos_count),
    ], dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Add weight decay for L2 regularization
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    def evaluate(loader):
        total_loss = 0.0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / max(1, len(loader))
        accuracy = 100.0 * correct / max(1, total)
        return avg_loss, accuracy

    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            running_total += labels.size(0)
            running_correct += (predicted == labels).sum().item()

        avg_loss = running_loss / max(1, len(train_loader))
        train_accuracy = 100.0 * running_correct / max(1, running_total)
        train_loss = avg_loss
        val_loss, val_accuracy = evaluate(val_loader)
        
        # Save best model
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_loss:.4f} | train acc: {train_accuracy:.2f}% | "
            f"val loss: {val_loss:.4f} | val acc: {val_accuracy:.2f}%"
        )

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model_path = ROOT.parent / "cinder_model.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved as {model_path.name} (best val acc: {best_val_acc:.2f}%)")


def load_trained_model(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m = build_model().to(device)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    return m, device


def predict_image(img_path, model_path):
    m, device = load_trained_model(model_path)
    tf = make_eval_transform()
    img = load_image(Path(img_path))
    inp = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = m(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    label = 'cancerous' if pred == 1 else 'non-cancerous'
    print(f'Prediction: {label} (class={pred}) probs={probs.tolist()}')
    return pred, probs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training')
    parser.add_argument('--predict', type=str, help='Path to image to predict')
    parser.add_argument('--model', type=str, default=str(ROOT.parent / 'cinder_model.pth'), help='Trained model path')
    parser.add_argument('--batch', action='store_true', help='Run batch prediction over datasets/raw')
    parser.add_argument('--aggregate', action='store_true', help='Aggregate predictions per-uuid and save CSV')
    parser.add_argument('--show', type=str, help='Show image preview and probs for a path')
    parser.add_argument('--eval', action='store_true', help='Evaluate model on labeled CSV')
    parser.add_argument('--all', action='store_true', help='Run batch, aggregate, show(sample), and eval')
    args = parser.parse_args()

    if args.train:
        train()
    elif args.predict:
        predict_image(args.predict, args.model)
    elif args.batch or args.all:
        # batch predict all images under RAW_DIR
        m, device = load_trained_model(args.model)
        tf = make_eval_transform()
        imgs = find_images(RAW_DIR)
        out_rows = []
        batch = []
        batch_paths = []
        bs = 32
        for p in imgs:
            batch.append(tf(load_image(Path(p))))
            batch_paths.append(str(p))
            if len(batch) >= bs:
                xb = torch.stack(batch).to(device)
                with torch.no_grad():
                    out = m(xb)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                for path, pr in zip(batch_paths, probs):
                    pred = int(pr.argmax())
                    out_rows.append((path, pred, float(pr[0]), float(pr[1])))
                batch = []
                batch_paths = []
        if len(batch) > 0:
            xb = torch.stack(batch).to(device)
            with torch.no_grad():
                out = m(xb)
                probs = torch.softmax(out, dim=1).cpu().numpy()
            for path, pr in zip(batch_paths, probs):
                pred = int(pr.argmax())
                out_rows.append((path, pred, float(pr[0]), float(pr[1])))

        import csv
        out_path = ROOT.parent / 'datasets' / 'predictions.csv'
        out_path.parent.mkdir(exist_ok=True)
        with open(out_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path','pred','prob0','prob1'])
            w.writerows(out_rows)
        print(f'Wrote {len(out_rows)} predictions to {out_path}')

        if args.all:
            # show first sample preview
            sample_path = out_rows[0][0]
            print('Sample prediction for', sample_path)
            predict_image(sample_path, args.model)

        if not args.aggregate and not args.eval:
            # finished if no further steps requested
            pass
    if args.aggregate or args.all:
        # read predictions and aggregate by uuid (parent folder)
        import pandas as pd
        preds = pd.read_csv(ROOT.parent / 'datasets' / 'predictions.csv')
        preds['uuid'] = preds['path'].apply(lambda p: Path(p).parent.name)
        agg = preds.groupby('uuid').agg({'prob1': ['mean'], 'pred': ['sum', 'count']})
        agg.columns = ['prob1_mean','pred_sum','count']
        # majority vote: pred_sum > count/2
        agg['majority_pred'] = (agg['pred_sum'] > (agg['count']/2)).astype(int)
        out_agg = ROOT.parent / 'datasets' / 'predictions_by_uuid.csv'
        agg.to_csv(out_agg)
        print(f'Wrote per-uuid aggregation to {out_agg}')

    if args.show:
        predict_image(args.show, args.model)

    if args.eval or args.all:
        # evaluate on labeled CSV using resolved label paths
        frame, cols = resolve_labeled_images()
        if frame is None:
            print('No labeled CSV found to evaluate.')
        else:
            path_col, label_col = cols
            df = frame[[path_col, label_col]].copy()
            # load model
            m, device = load_trained_model(args.model)
            tf = make_eval_transform()
            y_true = []
            y_pred = []
            y_prob1 = []
            batch = []
            batch_paths = []
            for idx, row in df.iterrows():
                p = row[path_col]
                batch.append(tf(load_image(Path(p))))
                batch_paths.append((p, int(row[label_col])))
                if len(batch) >= 32:
                    xb = torch.stack(batch).to(device)
                    with torch.no_grad():
                        out = m(xb)
                        probs = torch.softmax(out, dim=1).cpu().numpy()
                    for (pp, yy), pr in zip(batch_paths, probs):
                        y_true.append(yy)
                        y_pred.append(int(pr.argmax()))
                        y_prob1.append(float(pr[1]))
                    batch = []
                    batch_paths = []
            if len(batch) > 0:
                xb = torch.stack(batch).to(device)
                with torch.no_grad():
                    out = m(xb)
                    probs = torch.softmax(out, dim=1).cpu().numpy()
                for (pp, yy), pr in zip(batch_paths, probs):
                    y_true.append(yy)
                    y_pred.append(int(pr.argmax()))
                    y_prob1.append(float(pr[1]))

            import numpy as np
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            tp = int(((y_true == 1) & (y_pred == 1)).sum())
            tn = int(((y_true == 0) & (y_pred == 0)).sum())
            fp = int(((y_true == 0) & (y_pred == 1)).sum())
            fn = int(((y_true == 1) & (y_pred == 0)).sum())
            accuracy = 100.0 * (tp + tn) / max(1, tp + tn + fp + fn)
            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(1e-8, (precision + recall))
            print('Evaluation on labeled CSV:')
            print(f' TP={tp} TN={tn} FP={fp} FN={fn}')
            print(f' Accuracy={accuracy:.2f}% Precision={precision:.4f} Recall={recall:.4f} F1={f1:.4f}')
    else:
        parser.print_help()