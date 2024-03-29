from PIL import Image
import torch
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import optim
from model import Segmentor
from segmentation import SegmentationDataset, train, validation

img_transform = transforms.Compose([
    transforms.Resize((14*32,14*32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

mask_transform = transforms.Compose([
    transforms.Resize((32*4,32*4)),
    transforms.ToTensor(),
])

dataset = SegmentationDataset(img_dir=r"inputs/data/classification/train", mask_dir=r"inputs/data/segmentation/train", num_classes = 2, img_transform=img_transform, mask_transform=mask_transform)

# Splitting data into train and validation sets
train_imgs, valid_imgs = train_test_split(dataset.images, test_size=0.2, random_state=42)
train_masks, valid_masks = train_test_split(dataset.masks, test_size=0.2, random_state=42)

train_dataset = SegmentationDataset(img_dir=r"inputs/data/classification/train", mask_dir=r"inputs/data/segmentation/train", num_classes = 2, img_transform=img_transform, mask_transform=mask_transform, images=train_imgs, masks=train_masks)
valid_dataset = SegmentationDataset(img_dir=r"inputs/data/classification/train", mask_dir=r"inputs/data/segmentation/train", num_classes = 2, img_transform=img_transform, mask_transform=mask_transform, images=valid_imgs, masks=valid_masks)

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True) #, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=2) #, num_workers=4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Segmentor(2)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    train(model, train_loader, criterion, optimizer, epoch)
    validation(model, criterion, valid_loader)

torch.save(model.state_dict(), 'segmentation_model.pt')

