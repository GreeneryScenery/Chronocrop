# Import necessary modules
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import copy

class CustomImageFolder(DatasetFolder):
    def __init__(self, root, loader, extensions=None, transform=None, target_transform=None):
        super().__init__(root, loader, extensions, transform, target_transform)
        self.class_encoder = LabelEncoder()
        self.subclass_encoder = LabelEncoder()

        # Collect class and subclass labels
        class_labels = []
        subclass_labels = []
        for path, _ in self.samples:
            path = Path(path)
            class_labels.append(path.parent.parent.name)
            subclass_labels.append(path.parent.name)

        # Fit the encoders
        self.class_encoder.fit(class_labels)
        self.subclass_encoder.fit(subclass_labels)

    def __getitem__(self, index):
        path, _ = self.samples[index]
        sample = Image.open(path)
        path = Path(path)
        class_label = self.class_encoder.transform([path.parent.parent.name])[0]
        subclass_label = self.subclass_encoder.transform([path.parent.name])[0]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, class_label, subclass_label

def load_classification_data(data_dir='inputs/data/classification/train'):
    # Define transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the dataset from directory and apply transformations
    full_dataset = CustomImageFolder(data_dir, read_image, extensions=('.jpg', '.png'), transform=transform)

    # Split the full dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))  # 80% of the dataset is used for training
    val_size = len(full_dataset) - train_size  # The rest is used for validation

    # Randomly split dataset into training and validation dataset
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create data loaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle the training data
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

    # Create dictionaries that map class and subclass names to labels
    class_to_idx = {name: label for label, name in enumerate(full_dataset.class_encoder.classes_)}
    subclass_to_idx = {name: label for label, name in enumerate(full_dataset.subclass_encoder.classes_)}

    return {'train': train_loader, 'val': val_loader}, len(full_dataset.class_encoder.classes_), len(full_dataset.subclass_encoder.classes_), class_to_idx, subclass_to_idx

# # Define a function for loading and transforming image data
# def load_classification_data(data_dir='data/classification/train'):
#     # Define transformations: random crop, random flip, convert to tensor, and normalize
#     transform = transforms.Compose([
#         transforms.RandomResizedCrop(224),  # Resize and crop the image to a 224x224 square
#         transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
#         transforms.ToTensor(),  # Convert the image to a tensor
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image with mean and standard deviation
#     ])

#     # Load the dataset from directory and apply transformations
#     full_dataset = datasets.ImageFolder(data_dir, transform)

#     # Split the full dataset into train and validation sets
#     train_size = int(0.8 * len(full_dataset))  # 80% of the dataset is used for training
#     val_size = len(full_dataset) - train_size  # The rest is used for validation

#     # Randomly split dataset into training and validation dataset
#     train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

#     # Create data loaders for train and validation sets
#     # They provide an easy way to iterate over the dataset in mini-batches
#     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Shuffle the training data
#     val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # No need to shuffle validation data

#     return {'train': train_loader, 'val': val_loader}, len(full_dataset.classes)  # Return loaders and number of classes in the dataset

# Define a function to train the model
def train_classification_model(model, dataloaders, criterion, optimizer, scheduler, device, n_epochs_stop=6):
    # Initialize variables
    min_val_loss = np.Inf  # Minimum validation loss starts at infinity
    epochs_no_improve = 0  # No improvement in epochs counter

    # Loop over epochs
    for epoch in range(100):
        print('Epoch {}/{}'.format(epoch, 100 - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            # Initialize metrics for this phase
            running_loss = 0.0  # Accumulate losses over the epoch
            correct = 0  # Count correct predictions
            total = 0  # Count total predictions

            # Use tqdm for progress bar
            with tqdm(total=len(dataloaders[phase]), unit='batch') as p:
                for inputs, class_labels, subclass_labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    class_labels = class_labels.to(device)
                    subclass_labels = subclass_labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        class_outputs, subclass_outputs = model(inputs)
                        class_loss = criterion(class_outputs, class_labels)
                        subclass_loss = criterion(subclass_outputs, subclass_labels)
                        loss = class_loss + subclass_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    correct += (class_outputs.argmax(dim=1) == class_labels).sum().item()
                    total += class_labels.size(0)

                    # Update the progress bar
                    p.set_postfix({'loss': loss.item(), 'accuracy': 100 * correct / total})
                    p.update(1)

                # Calculate loss and accuracy for this epoch
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = 100 * correct / total

                print('{} Loss: {:.4f} Acc: {:.2f}%'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_loss < min_val_loss:
                print(f'Validation Loss Decreased({min_val_loss:.6f}--->{epoch_loss:.6f}) \t Saving The Model')
                min_val_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve == n_epochs_stop:
                    print('Early stopping!')
                    model.load_state_dict(best_model_wts)
                    return model

        scheduler.step(epoch_loss)

    return model