# ========== 数据处理精简交付脚本（仅模型训练使用，无绘图/校验冗余代码） ==========
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

# 自动区分Kaggle云端 / 本地运行环境
if os.path.exists("/kaggle/input"):
    DATA_ROOT = "/kaggle/input/competitions/bh-2026/img_AIdet/data"
else:
    # 本地运行时仅需修改此处根目录
    DATA_ROOT = "D:/bh-2026/img_AIdet/data"

# 全局固定超参
TRAIN_REAL = os.path.join(DATA_ROOT, "train", "REAL")
TRAIN_FAKE = os.path.join(DATA_ROOT, "train", "FAKE")
TEST_DIR = os.path.join(DATA_ROOT, "test")
IMG_SIZE = 32
BATCH_SIZE = 512
SPLIT_SEED = 42

# 训练集数据集读取类
class TrainAIDataset(Dataset):
    def __init__(self, real_dir, fake_dir, transform=None):
        self.transform = transform
        self.samples = []
        # 优先加载真实图片，标签0
        for fname in sorted(os.listdir(real_dir)):
            if fname.endswith(".jpg"):
                file_path = os.path.join(real_dir, fname)
                self.samples.append((file_path, 0))
        # 后加载AI假图，标签1
        for fname in sorted(os.listdir(fake_dir)):
            if fname.endswith(".jpg"):
                file_path = os.path.join(fake_dir, fname)
                self.samples.append((file_path, 1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label

# 测试集数据集读取类（输出图片ID用于生成提交csv）
class TestAIDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.transform = transform
        self.test_dir = test_dir
        # 测试图片固定编号1~20000
        self.file_list = [(i, f"{i}.jpg") for i in range(1, 20001)]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_id, fname = self.file_list[idx]
        img_path = os.path.join(self.test_dir, fname)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, img_id

# 训练集专属数据增强（严格遵循作业要求）
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 8, IMG_SIZE + 8)),
    transforms.RandomCrop(IMG_SIZE, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集、测试集变换：无随机增强，仅标准化缩放
val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载完整训练数据集
full_train_dataset = TrainAIDataset(TRAIN_REAL, TRAIN_FAKE, train_transform)
# 固定seed拆分9:1训练/验证集
split_generator = torch.Generator().manual_seed(SPLIT_SEED)
val_sample_num = int(len(full_train_dataset) * 0.1)
train_sample_num = len(full_train_dataset) - val_sample_num
train_dataset, val_dataset = random_split(full_train_dataset, [train_sample_num, val_sample_num], generator=split_generator)
# 验证集替换为无增强transform
val_dataset.dataset.transform = val_test_transform

# 加载测试集
test_dataset = TestAIDataset(TEST_DIR, val_test_transform)

# 批量加载器（适配Kaggle num_workers=0，本地可自行修改为2/4加速）
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
