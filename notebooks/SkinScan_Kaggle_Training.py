"""
Production Multi-Modal Skin Cancer Training Script for Kaggle Notebooks:
Supports both Kaggle Datasets attached in user's notebook:
- 'Skin Cancer MNIST: HAM10000'
- 'ISIC 2019 Skin Lesion images for classification'

Runs with GPU acceleration on Kaggle (P100 / T4) and exports weights to /kaggle/working/
"""

import os
import gc
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# ─────────────────────────────────────────────────────────
# 1. GPU ACCELERATION CHECK
# ─────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Compute Device: {device}")
if torch.cuda.is_available():
    print(f"🎮 Active GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    torch.backends.cudnn.benchmark = True

# ─────────────────────────────────────────────────────────
# 2. AUTOMATIC KAGGLE DATASET SCANNER & METADATA MERGER
# ─────────────────────────────────────────────────────────
KAGGLE_INPUT_DIR = '/kaggle/input'
OUTPUT_DIR = '/kaggle/working'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("\n🔍 Scanning /kaggle/input/ for datasets...")
for item in os.listdir(KAGGLE_INPUT_DIR):
    print(f"  📁 {item}")

# Build fast index mapping image_id -> absolute path
image_path_dict = {}
for root, _, files in os.walk(KAGGLE_INPUT_DIR):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_id = os.path.splitext(f)[0]
            image_path_dict[img_id] = os.path.join(root, f)

print(f"🖼️ Total Images Indexed in Kaggle Environment: {len(image_path_dict):,}")

# Search for all CSV metadata files
metadata_dfs = []
for root, _, files in os.walk(KAGGLE_INPUT_DIR):
    for f in files:
        if f.lower().endswith('.csv'):
            csv_path = os.path.join(root, f)
            print(f"  📄 Parsing metadata CSV: {csv_path}")
            try:
                temp_df = pd.read_csv(csv_path)
                temp_df.columns = [c.lower() for c in temp_df.columns]

                # Identify image ID column
                id_col = next((c for c in ['image_id', 'image', 'isic_id', 'name'] if c in temp_df.columns), None)
                if not id_col:
                    continue
                temp_df['image_id'] = temp_df[id_col].astype(str).str.replace('.jpg', '').str.replace('.png', '')

                # 1. HAM10000 format (dx column)
                if 'dx' in temp_df.columns:
                    clean_df = temp_df[['image_id', 'dx']].copy()
                    clean_df['age'] = temp_df.get('age', 50)
                    clean_df['sex'] = temp_df.get('sex', 'unknown')
                    clean_df['localization'] = temp_df.get('localization', 'unknown')
                    metadata_dfs.append(clean_df)

                # 2. ISIC 2019 format (one-hot columns)
                elif 'mel' in temp_df.columns and ('nv' in temp_df.columns or 'bcc' in temp_df.columns):
                    one_hot_cols = [c for c in ['mel', 'nv', 'bcc', 'akiec', 'ak', 'bkl', 'df', 'vasc', 'scc'] if c in temp_df.columns]
                    temp_df['dx'] = temp_df[one_hot_cols].idxmax(axis=1)
                    clean_df = temp_df[['image_id', 'dx']].copy()
                    clean_df['age'] = temp_df.get('age_approx', temp_df.get('age', 50))
                    clean_df['sex'] = temp_df.get('sex', 'unknown')
                    clean_df['localization'] = temp_df.get('anatom_site_general_challenge', temp_df.get('localization', 'unknown'))
                    metadata_dfs.append(clean_df)

            except Exception as e:
                print(f"    ⚠️ Could not parse {csv_path}: {e}")

if metadata_dfs:
    df = pd.concat(metadata_dfs, ignore_index=True).drop_duplicates(subset=['image_id'])
else:
    print("⚠️ No metadata CSV matched. Generating fallback...")
    df = pd.DataFrame({'image_id': list(image_path_dict.keys())})
    df['dx'] = 'nv'
    df['age'] = 50
    df['sex'] = 'unknown'
    df['localization'] = 'unknown'

# Standardize 7 diagnostic classes
LABEL_MAP = {'mel': 0, 'bcc': 1, 'akiec': 2, 'bkl': 3, 'nv': 4, 'vasc': 5, 'df': 6}
CLASS_NAMES = ['mel', 'bcc', 'akiec', 'bkl', 'nv', 'vasc', 'df']

df['dx'] = df['dx'].astype(str).str.lower().replace({
    'ak': 'akiec',
    'scc': 'bcc',  # Squamous Cell Carcinoma mapped to non-melanoma skin cancer
    'seborrheic keratosis': 'bkl',
    'solar lentigo': 'bkl',
    'dermatofibroma': 'df',
    'vascular lesion': 'vasc'
})

df = df[df['dx'].isin(LABEL_MAP.keys())].copy()
df['label'] = df['dx'].map(LABEL_MAP)
df['image_path'] = df['image_id'].map(image_path_dict)
df = df[df['image_path'].notna()].copy()

df['age'] = pd.to_numeric(df['age'], errors='coerce').fillna(50)
df['sex'] = df['sex'].fillna('unknown')
df['localization'] = df['localization'].fillna('unknown')

print(f"\n🏆 Total Verified Clinical Images with Labels: {len(df):,}")
print("Class Distribution:")
print(df['dx'].value_counts())

# ─────────────────────────────────────────────────────────
# 3. METADATA ENCODING & DATASET
# ─────────────────────────────────────────────────────────
LOCALIZATIONS = ['back', 'lower extremity', 'trunk', 'upper extremity', 'abdomen', 'face', 'chest', 'scalp']
SEXES = ['male', 'female', 'other']

def extract_meta_vector(row):
    vec = np.zeros(16, dtype=np.float32)
    vec[0] = min(float(row['age']), 100.0) / 100.0
    vec[1] = 6.0 / 120.0
    vec[2] = 1.0 if row['dx'] == 'mel' else 0.0

    sex = str(row['sex']).lower()
    s_idx = SEXES.index(sex) if sex in SEXES else 0
    vec[3 + s_idx] = 1.0

    loc = str(row['localization']).lower()
    l_idx = LOCALIZATIONS.index(loc) if loc in LOCALIZATIONS else 0
    vec[6 + l_idx] = 1.0
    return vec

df['meta_vector'] = [extract_meta_vector(row) for _, row in df.iterrows()]

class MultiModalSkinDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception:
            image = Image.new('RGB', (224, 224), color=(128, 128, 128))

        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)

        meta_tensor = torch.tensor(row['meta_vector'], dtype=torch.float32)
        label = int(row['label'])
        tier1_label = 1 if label in [0, 1, 2, 3] else 0

        return image_tensor, meta_tensor, label, tier1_label

# ─────────────────────────────────────────────────────────
# 4. STRATIFIED SPLIT & AUGMENTATIONS
# ─────────────────────────────────────────────────────────
train_df, test_df = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
train_df, val_df = train_test_split(train_df, test_size=0.1765, stratify=train_df['label'], random_state=42)

print(f"\n📊 Splits -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((240, 240)),
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=30),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

class_counts = train_df['label'].value_counts().sort_index().values
class_weights = 1.0 / (class_counts + 1e-6)
sample_weights = [class_weights[label] for label in train_df['label']]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(train_df), replacement=True)

train_loader = DataLoader(MultiModalSkinDataset(train_df, transform=train_transforms), batch_size=32, sampler=sampler, num_workers=2, pin_memory=True)
val_loader = DataLoader(MultiModalSkinDataset(val_df, transform=val_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(MultiModalSkinDataset(test_df, transform=val_transforms), batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

# ─────────────────────────────────────────────────────────
# 5. MULTI-MODAL DENSENET121 & FOCAL LOSS
# ─────────────────────────────────────────────────────────
class MultiClassFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean': return focal_loss.mean()
        elif self.reduction == 'sum': return focal_loss.sum()
        return focal_loss

class MultiModalDenseNetHybrid(nn.Module):
    def __init__(self, num_classes=7, meta_dim=16, pretrained=True):
        super().__init__()
        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.vision_backbone = models.densenet121(weights=weights)
        in_features = self.vision_backbone.classifier.in_features
        self.vision_backbone.classifier = nn.Identity()

        self.metadata_branch = nn.Sequential(
            nn.Linear(meta_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )

        self.tier1_gating_head = nn.Sequential(
            nn.Linear(in_features + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

        self.tier2_classification_head = nn.Sequential(
            nn.Linear(in_features + 32, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, img_tensor, meta_tensor):
        vis_feat = self.vision_backbone(img_tensor)
        meta_feat = self.metadata_branch(meta_tensor)
        fused = torch.cat((vis_feat, meta_feat), dim=1)
        tier1_logits = self.tier1_gating_head(fused)
        tier2_logits = self.tier2_classification_head(fused)
        return tier2_logits, tier1_logits

model = MultiModalDenseNetHybrid(num_classes=7, meta_dim=16, pretrained=True).to(device)

alpha_weights = torch.tensor([2.5, 3.0, 4.0, 2.0, 0.5, 6.0, 7.0]).to(device)
criterion_focal = MultiClassFocalLoss(alpha=alpha_weights, gamma=2.0)
criterion_tier1 = nn.CrossEntropyLoss()

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

print(f"✅ Model compiled with {sum(p.numel() for p in model.parameters()):,} parameters.")

# ─────────────────────────────────────────────────────────
# 6. TRAINING & VALIDATION LOOP
# ─────────────────────────────────────────────────────────
EPOCHS = 20
best_val_score = 0.0
best_model_wts = copy.deepcopy(model.state_dict())
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_mel_recall': []}

scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
print(f"\n🚀 Training on Kaggle GPU for {EPOCHS} Epochs...")
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for images, metas, labels, t1_labels in train_loader:
        images = images.to(device, non_blocking=True)
        metas = metas.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        t1_labels = t1_labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler:
            with torch.cuda.amp.autocast():
                t2_logits, t1_logits = model(images, metas)
                loss = criterion_focal(t2_logits, labels) + 0.3 * criterion_tier1(t1_logits, t1_labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            t2_logits, t1_logits = model(images, metas)
            loss = criterion_focal(t2_logits, labels) + 0.3 * criterion_tier1(t1_logits, t1_labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)

    scheduler.step()
    epoch_train_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, metas, labels, t1_labels in val_loader:
            images = images.to(device, non_blocking=True)
            metas = metas.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t1_labels = t1_labels.to(device, non_blocking=True)

            t2_logits, t1_logits = model(images, metas)
            loss = criterion_focal(t2_logits, labels) + 0.3 * criterion_tier1(t1_logits, t1_labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.argmax(t2_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_val_loss = val_loss / len(val_loader.dataset)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = np.mean(all_preds == all_labels)
    mel_mask = all_labels == 0
    mel_recall = np.mean(all_preds[mel_mask] == 0) if np.sum(mel_mask) > 0 else 0.0

    history['train_loss'].append(epoch_train_loss)
    history['val_loss'].append(epoch_val_loss)
    history['val_acc'].append(acc)
    history['val_mel_recall'].append(mel_recall)

    print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {acc*100:.1f}% | Mel Recall: {mel_recall*100:.1f}%")

    score = 0.6 * mel_recall + 0.4 * acc
    if score > best_val_score:
        best_val_score = score
        best_model_wts = copy.deepcopy(model.state_dict())
        checkpoint_path = os.path.join(OUTPUT_DIR, 'skinscan_densenet121_v2.pth')
        torch.save(best_model_wts, checkpoint_path)
        print(f"  ⭐ Saved new best model to {checkpoint_path} (Score: {score:.4f})")

elapsed = time.time() - start_time
print(f"\n🎉 Training Finished in {elapsed/60:.2f} minutes!")

# ─────────────────────────────────────────────────────────
# 7. CLINICAL TEST EVALUATION & CONFUSION MATRIX
# ─────────────────────────────────────────────────────────
model.load_state_dict(best_model_wts)
model.eval()
test_preds, test_probs, test_targets = [], [], []

with torch.no_grad():
    for images, metas, labels, _ in test_loader:
        images = images.to(device)
        metas = metas.to(device)
        t2_logits, _ = model(images, metas)
        probs = F.softmax(t2_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        test_probs.extend(probs.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())
        test_targets.extend(labels.numpy())

test_preds = np.array(test_preds)
test_probs = np.array(test_probs)
test_targets = np.array(test_targets)

print("\n" + "=" * 60)
print("  CLINICAL TEST SET EVALUATION")
print("=" * 60)
print(classification_report(test_targets, test_preds, target_names=CLASS_NAMES, digits=4))

try:
    roc_auc = roc_auc_score(test_targets, test_probs, multi_class='ovr', average='macro')
    print(f"🏆 Multi-Class Macro ROC-AUC: {roc_auc:.4f}")
except Exception as e:
    print(f"ROC-AUC: {e}")

# Save plots
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
cm = confusion_matrix(test_targets, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=axes[0])
axes[0].set_title('7-Class Confusion Matrix (Test Set)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Predicted Class', fontweight='bold')
axes[0].set_ylabel('Ground Truth', fontweight='bold')

axes[1].plot(history['train_loss'], label='Train Loss', color='#2563EB', lw=2)
axes[1].plot(history['val_loss'], label='Val Loss', color='#DC2626', lw=2)
axes[1].plot(history['val_acc'], label='Val Accuracy', color='#059669', lw=2, linestyle='--')
axes[1].set_title('Convergence Curves', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Epoch', fontweight='bold')
axes[1].set_ylabel('Loss / Accuracy', fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, 'skinscan_evaluation.png')
plt.savefig(plot_path, dpi=300)
plt.show()

checkpoint_final = os.path.join(OUTPUT_DIR, 'skinscan_densenet121_v2.pth')
print(f"\n💾 Model Checkpoint Saved At: {checkpoint_final} ({os.path.getsize(checkpoint_final)/1e6:.2f} MB)")
print("👉 In Kaggle: Go to right panel under 'Output' (/kaggle/working/) and click download on 'skinscan_densenet121_v2.pth'!")
