import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.utils import resample
import pandas as pd
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Config
DATA_DIR = 'aptos2019/train_images'
CSV_PATH = 'aptos2019/train.csv'
SUBSET_PATH = 'aptos2019/train_subset.csv'  # file chứa subset
NUM_SAMPLES = None  # Sử dụng toàn bộ tập dữ liệu
IMG_SIZE = 224
LR = 5e-5
BATCH_SIZE = 32
EPOCHS = 50
NUM_CLASSES = 5

# Dataset
class DRDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0] + '.png'
        label = int(self.data.iloc[idx, 1])
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Tạo file subset nếu chưa có
if not os.path.exists(SUBSET_PATH):
    data = pd.read_csv(CSV_PATH)
    print('Label distribution in original data:')
    print(data['diagnosis'].value_counts())
    # Cân bằng nhãn bằng oversampling
    dfs = []
    max_count = data['diagnosis'].value_counts().max()
    for label in data['diagnosis'].unique():
        df_label = data[data['diagnosis'] == label]
        dfs.append(resample(df_label, replace=True, n_samples=max_count, random_state=42))
    data_balanced = pd.concat(dfs)
    data_balanced = data_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    print('Label distribution after balancing:')
    print(data_balanced['diagnosis'].value_counts())
    # Nếu NUM_SAMPLES là None, lấy toàn bộ data_balanced
    if NUM_SAMPLES is not None:
        data_balanced = data_balanced.sample(n=NUM_SAMPLES, random_state=42).reset_index(drop=True)
    data_balanced.to_csv(SUBSET_PATH, index=False)

# Tách train/val (80/20)
TRAIN_SPLIT_PATH = 'aptos2019/train_split.csv'
VAL_SPLIT_PATH = 'aptos2019/val_split.csv'
if not (os.path.exists(TRAIN_SPLIT_PATH) and os.path.exists(VAL_SPLIT_PATH)):
    subset = pd.read_csv(SUBSET_PATH)
    train_df = subset.sample(frac=0.8, random_state=42)
    val_df = subset.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    train_df.to_csv(TRAIN_SPLIT_PATH, index=False)
    val_df.to_csv(VAL_SPLIT_PATH, index=False)


# Train/Val dataset & dataloader
train_dataset = DRDataset(TRAIN_SPLIT_PATH, DATA_DIR, transform)
val_dataset = DRDataset(VAL_SPLIT_PATH, DATA_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# Model
from torchvision.models import EfficientNet_B3_Weights
# Sử dụng EfficientNet-B3
from torchvision.models import efficientnet_b3
model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.cuda() if torch.cuda.is_available() else model

# Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=5e-5)

# Training loop
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    # EarlyStopping & ModelCheckpoint
    class EarlyStopping:
        def __init__(self, patience=7, min_delta=0.001):
            self.patience = patience
            self.min_delta = min_delta
            self.counter = 0
            self.best_score = None
            self.early_stop = False
        def __call__(self, val_acc):
            if self.best_score is None:
                self.best_score = val_acc
            elif val_acc < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = val_acc
                self.counter = 0

    best_val_acc = 0.0
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    for epoch in range(EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.cuda() if torch.cuda.is_available() else images
            labels = labels.cuda() if torch.cuda.is_available() else labels
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / total
        epoch_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.cuda() if torch.cuda.is_available() else images
                labels = labels.cuda() if torch.cuda.is_available() else labels
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        train_accs.append(epoch_acc)
        val_accs.append(val_acc)

        print(f'Epoch {epoch+1}/{EPOCHS} - Train Loss: {epoch_loss:.4f} - Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}')

        # Lưu mô hình tốt nhất
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'dr_best_model.pth')
            print(f'Best model saved at epoch {epoch+1} with Val Acc: {val_acc:.4f}')

        # Early stopping
        early_stopping(val_acc)
        if early_stopping.early_stop:
            print('Early stopping triggered.')
            break

    print(f'Best Val Acc: {best_val_acc:.4f}')

    # Vẽ loss/accuracy quá trình train/val
    fig, axs = plt.subplots(1, 2, figsize=(18,6))
    axs[0].plot(train_losses, label='Train Loss')
    axs[0].plot(val_losses, label='Val Loss')
    axs[0].legend()
    axs[0].set_title('Loss Curve EfficientNet_B3')
    axs[1].plot(train_accs, label='Train Accuracy')
    axs[1].plot(val_accs, label='Val Accuracy')
    axs[1].legend()
    axs[1].set_title('Accuracy Curve EfficientNet_B3')
    plt.tight_layout()
    plt.savefig('training_curves.png')
    print('Đã lưu ảnh trực quan quá trình huấn luyện vào file training_curves.png')
    # Nếu môi trường hỗ trợ hiển thị, show ảnh luôn
    try:
        get_ipython
        plt.show()
    except Exception:
        pass
    print('Nếu bạn chạy file .py trên terminal, hãy mở file training_curves.png để xem biểu đồ. Nếu chạy trong notebook, ảnh sẽ hiện trực tiếp.')
