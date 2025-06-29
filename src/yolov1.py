import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import torchvision.transforms as T

class SimpleYOLOv1(nn.Module):
    def __init__(self, num_classes=5, S=7, B=2):
        super(SimpleYOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = num_classes
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * (S * S), 4096),
            nn.ReLU(),
            nn.Linear(4096, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x.view(-1, self.S, self.S, self.B * 5 + self.C)

class TACOSubset(Dataset):
    def __init__(self, img_dir, label_dir, transforms=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.imgs = [f for f in os.listdir(img_dir) if f.endswith('.jpg')]
        self.transforms = transforms

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        label_path = os.path.join(self.label_dir, self.imgs[idx].replace('.jpg', '.txt'))
        img = Image.open(img_path).convert("RGB")
        with open(label_path, 'r') as f:
            labels = [list(map(float, line.split())) for line in f]
        boxes = torch.tensor([[l[1], l[2], l[3], l[4]] for l in labels], dtype=torch.float32)
        class_ids = torch.tensor([int(l[0]) for l in labels], dtype=torch.int64)
        if self.transforms:
            img = self.transforms(img)
        return img, {'boxes': boxes, 'labels': class_ids}

    def __len__(self):
        return len(self.imgs)

dataset = TACOSubset(
    img_dir='/home/hanna/course_project/data/TACO/yolo_dataset/train/images',
    label_dir='/home/hanna/course_project/data/TACO/yolo_dataset/train/labels',
    transforms=T.ToTensor()
)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = SimpleYOLOv1(num_classes=5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(50):
    for images, targets in loader:
        images = images.cuda()
        outputs = model(images)
        # Placeholder loss (implement YOLOv1 loss)
        loss = torch.mean(outputs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
torch.save(model.state_dict(), 'runs/detect/yolov1_taco.pt')