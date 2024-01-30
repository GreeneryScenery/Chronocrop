from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import os
from tqdm import tqdm
import numpy as np
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def replace_colors_with_integers(image):
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary image, all black pixels will be 0 and all non-black pixels will be 1
    _, binary_image = cv2.threshold(grayscale_image, 1, 1, cv2.THRESH_BINARY)

    return binary_image

class SegmentationDataset(Dataset):
    def __init__(self, img_dir, mask_dir, num_classes, img_transform=None, mask_transform=None, images=None, masks=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes

        # Only include images for which a mask is found
        if images is None:
            self.images = []
            self.masks = []
            for path, _, files in os.walk('inputs/data/classification/train'):
                for name in files:
                    target = path.replace('inputs/data/classification/train', 'inputs/data/segmentation/train') + '/' + name.split(".")[0] + ".png"
                    if os.path.isfile(target):
                        self.images.append(path + '/' + name)
                        self.masks.append(target)
            # self.images = [img for img in os.listdir(img_dir) if os.path.isfile(os.path.join(mask_dir, img.split(".")[0] + ".png"))]
        else:
            self.images = images
            self.masks = masks


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # img_name = self.images[idx].split(".")[0]
        # img_path = os.path.join(self.img_dir, self.images[idx])
        # mask_path = os.path.join(self.mask_dir, img_name + ".png")
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.fromarray(replace_colors_with_integers(np.array(Image.open(mask_path).convert("RGB"))))

        if self.img_transform:
            image = self.img_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask) * 255
        # Create a tensor to hold the binary masks
        bin_mask = torch.zeros(self.num_classes, mask.shape[1], mask.shape[2])

        # Ensure mask is a torch tensor and is in the same device as bin_mask
        mask = torch.from_numpy(np.array(mask)).to(bin_mask.device)
        
        # Convert mask to type float for comparison
        mask = mask.float()

        for i in range(self.num_classes):
            bin_mask[i] = (mask == i).float()  # Ensure resulting mask is float type

        return image, bin_mask

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    loop = tqdm(train_loader, total=len(train_loader))
    running_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(loop):
        # print(batch_idx) 
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss = loss.item())

    print(f'\nTrain set: Average loss: {running_loss/len(train_loader):.4f}')

def validation(model, criterion, valid_loader):
    model.eval()
    running_loss = 0
    correct = 0

    with torch.no_grad():
        loop = tqdm(valid_loader, total=len(valid_loader))
        for data, target in loop:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)

    print(f'\nValidation set: Average loss: {running_loss/len(valid_loader):.4f}')

def infer(image, model, device, img_transform):
    transformed_image = img_transform(image).unsqueeze(0).to(device)  # Add batch dimension and move to device

    # Make sure the model is in evaluation mode
    model.eval()

    with torch.no_grad():
        # Make prediction
        output = model(transformed_image)

        # Get the predicted class for each pixel
        _, predicted = torch.max(output, 1)
    
    # Move prediction to cpu and convert to numpy array
    predicted = predicted.squeeze().cpu().numpy()

    return transformed_image.cpu().squeeze().permute(1, 2, 0).numpy(), predicted