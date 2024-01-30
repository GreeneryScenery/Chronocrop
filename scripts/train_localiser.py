# Import necessary modules and functions
from localisation import load_localisation_data, train_localisation_model
from model import Localiser  # Import custom model from model.py file
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Check if CUDA is available and set PyTorch to use GPU or CPU accordingly
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Use the load_data function from tools.training to load our dataset
# This function presumably returns a set of data loaders and the number of classes in the dataset
dataloaders, num_classes, class_to_idx = load_localisation_data()

idx_to_class = {v: k for k, v in class_to_idx.items()}

print(idx_to_class)

# Initialize our localiser model with the number of output classes equal to num_classes
model = Localiser(num_classes) # this will load the small model
# model = Localiser(num_classes, backbone = 'dinov2_b') # to load the base model
# model = Localiser(num_classes, backbone = 'dinov2_l') # to load the large model
# model = Localiser(num_classes, backbone = 'dinov2_g') # to load the largest model

# Move the model to the device (GPU or CPU)
model.to(device)

# Set our loss function to Binary Cross Entropy With Logits Loss, a common choice for multi-label classification problems
criterion = nn.BCEWithLogitsLoss()

# Initialize Stochastic Gradient Descent (SGD) as our optimizer
# Set the initial learning rate to 0.001 and momentum to 0.9
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Initialize a learning rate scheduler that reduces learning rate when a metric has stopped improving
# In this case, we're monitoring the minimum validation loss with a patience of 7 epochs 
# i.e., the learning rate will be reduced if the validation loss does not improve for 7 consecutive epochs
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=7, verbose=True)

# Finally, use the train_model function from tools.training to train our model
# The model, dataloaders, loss function, optimizer, learning rate scheduler, and device are passed as arguments
model = train_localisation_model(model, dataloaders, criterion, optimizer, scheduler, device)

torch.save(model.state_dict(), 'localisation_model.pt')