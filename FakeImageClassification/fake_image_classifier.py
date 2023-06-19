import os
import random
import numpy as np
import torch
import pandas as pd
# Pandas is a table library in python, we may want to consider which class has least images to determine how
# to train the model.

# Setting up GPU for model training.
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Variable used below to set model training mode to use GPU

# CUBLAS_WORKSPACE_CONFIG=:4096:8 needed for pytorch reproducibility
# https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# Set seed to ensure model reproducibility.
def set_seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed_all(1)

# Dataset Statistics
# Make sure to extract the zip folder into its own folder /CV_Workshop_Dateset
DATASET_ROOT = './CV_Workshop_Dataset'
# Show current path of directory for debugging:
# print(os.path.abspath('./'))
DATSET_TRAIN_FOLDER = os.path.join(DATASET_ROOT, 'train')
DATASET_TRAIN_CSV = os.path.join(DATASET_ROOT, 'train.csv')
DATSET_TEST_FOLDER = os.path.join(DATASET_ROOT, 'test')
DATASET_TEST_CSV = os.path.join(DATASET_ROOT, 'test.csv')

train_df = pd.read_csv(DATASET_TRAIN_CSV)
class_counts = train_df['class'].value_counts()
print(class_counts)
print(train_df)

# # Samples of images:
# import climage
#
# def PrintImageSamples(image_paths, label):
#     for i, path in enumerate(image_paths):
#         print(str(i) + ": " + str(label))
#         output = climage.convert(path)
#         print(output)
#
# SAMPLES = 8
# real_image_paths = train_df[train_df['class'] == 'real'][:SAMPLES]['filename']
# real_image_paths = [os.path.join(DATASET_ROOT, p) for p in real_image_paths]
# PrintImageSamples(real_image_paths, 'real')
#
# fake_image_paths = train_df[train_df['class'] == 'fake'][:SAMPLES]['filename']
# fake_image_paths = [os.path.join(DATASET_ROOT, p) for p in fake_image_paths]
# PrintImageSamples(fake_image_paths, 'fake')

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.io import read_image # Load image file path to ndarray

LABELS_MAP = { # Convert text labels to class index, model is trained using class indexes
    'real': 0 ,
    'fake': 1 }
NUM_CLASS = len(LABELS_MAP)

class ImageDatasetFromCSV(Dataset):
    def __init__(self, annotations_csv, root_dir, has_labels, transform=None):
        self.data_frame = pd.read_csv(annotations_csv)
        self.root_dir = root_dir
        self.has_labels = has_labels
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.data_frame.iloc[idx, 0]) # data_frame column 0 are filenames
        image = read_image(image_path) # In pytorch, image Tensor is in C H W dimensions
        if self.transform:
            image = self.transform(image) # Apply image augmentation transformations

        if self.has_labels: # Only training set has labels, set label=None for train=False
            label = self.data_frame.iloc[idx, 1] # Index 1 = class
            label_idx = LABELS_MAP[label] # Convert textual label to a class index
            return {"image": image, "label": label_idx}
        else:
            return {"image": image}

# Train set Transforms
# Models usually take in float32 (instead of uint8 [0..255])
TRAIN_IM_W = 244
TRAIN_IM_H = 244 # 299 for inception, 384 for efficientnetv2_s, 244 for most others.
train_transform = transforms.Compose([
        transforms.Resize((TRAIN_IM_W, TRAIN_IM_H)),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(dtype=torch.float32)])

# Transform data type from unit8 to a float bc of some error.
#Preprocessing techniques to increase the number of training datasets


# Test set Transforms
TEST_IM_W = 244
TEST_IM_H = 244
test_transform = transforms.Compose([
        transforms.Resize((TEST_IM_W, TEST_IM_H)),
        transforms.ConvertImageDtype(dtype=torch.float32)])

# Portion out some train images for validation
TRAIN_VALID_SPLIT = [0.8, 0.2] # Ratio of train/validation split

labelled_dataset = ImageDatasetFromCSV(annotations_csv=DATASET_TRAIN_CSV, root_dir=DATASET_ROOT, has_labels=True, transform=train_transform)
train_dataset, valid_dataset = random_split(labelled_dataset, TRAIN_VALID_SPLIT)
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

# Create dataset from testing unlabelled data. Set has_labels = False since no label in csv.
test_dataset = ImageDatasetFromCSV(annotations_csv=DATASET_TEST_CSV, root_dir=DATASET_ROOT, has_labels=False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Dataset statistics
print(f"Total Samples - Train:{len(train_dataset)}, Val:{len(valid_dataset)}, Test:{len(test_dataset)}")

# Train dataloader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from glob import glob


extracted_batch = next(iter(train_dataloader)) # Get first batch
print(f"Images batch shape: {extracted_batch['image'].size()}")
print(f"Labels batch shape: {extracted_batch['label'].size()}")
extracted_image = extracted_batch['image'][0].squeeze() # Take 1 image to visualize
extracted_label_idx = extracted_batch['label'][0] # Take corresponding label index of image
label = list(LABELS_MAP.keys())[list(LABELS_MAP.values()).index(extracted_label_idx)] # Convert label index back to text
print(f"Label Index: {extracted_label_idx}, Label Class: {label}")

# rcParams['figure.figsize'] = 4,4 # Change display figure size
# plt.imshow(np.transpose(extracted_image, (1, 2, 0))) # Convert pytorch C H W to numpy image H W C format
plt.show()

# Test dataloader
extracted_test_batch = next(iter(test_dataloader))
# print(f"Images batch shape: {extracted_test_batch['image'].size()}")
extracted_image = extracted_test_batch['image'][0].squeeze()
# plt.imshow(np.transpose(extracted_image, (1, 2, 0)))
plt.show()

from torch.nn import Linear
from torchvision.models import resnet50, ResNet50_Weights

### Change your model here, or define your custom model
def MyModel():
    pre_trained = ResNet50_Weights.DEFAULT # Choose to use pre-trained weights or not (train from scratch)
    model = resnet50(weights=pre_trained)
    model.fc = Linear(in_features=2048, out_features = NUM_CLASS) # Change out last fully connected layer to match number of classes in dataset
    return model

model = MyModel()
print(model)


from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Adadelta, RMSprop # Optimizers

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=0.001, momentum=0.9) # Configure learning rate parameters as needed

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

early_stopper = EarlyStopper(patience=3, min_delta=0)

from tqdm import tqdm
from time import sleep


def train(model, train_dataloader, valid_dataloader, optimizer, criterion,
          num_epochs, device, save_model_path, early_stopper=None):
    model.train() # Set pytorch model to train mode
    model.to(device) # Set pytorch model to run in GPU or CPU mode

    for epoch in range(num_epochs):

        # Train
        running_train_loss = 0.0
        train_epoch = tqdm(train_dataloader) # tqdm = Progress bar
        for batch_idx, batch in enumerate(train_epoch):
            # Get batch of images & labels from Dataloader
            images = batch['image']
            labels = batch['label']
            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_train_loss += loss.item()
            train_avg_loss = running_train_loss/(batch_idx+1)
            train_epoch.set_description(f"Epoch {epoch}, Train loss: {train_avg_loss:.5f}")

        # Validation
        running_valid_loss = 0.0
        valid_epoch = tqdm(valid_dataloader)
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(valid_epoch):
            images = batch['image']
            labels = batch['label']
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_valid_loss += loss.item()
            valid_avg_loss = running_valid_loss/(batch_idx+1)
            valid_epoch.set_description(f"Epoch {epoch}, Valid loss: {valid_avg_loss:.5f}")

        print(f"Summary: Epoch {epoch} - Train loss: {train_avg_loss:.5f}, Valid loss: {valid_avg_loss:.5f}")

        # Save Model Checkpoint. See https://pytorch.org/tutorials/beginner/saving_loading_models.html#save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': valid_avg_loss,
            }, save_model_path)

        # Early Stopping
        if early_stopper is not None:
            if early_stopper.early_stop(valid_avg_loss):
                print(f"Early Stopping: No improvement for last {early_stopper.patience} epochs, stopping training")
                break

EPOCH = 3
SAVE_MODEL_PATH = 'model.pt'
train(model, train_dataloader, valid_dataloader, optimizer, criterion, EPOCH, DEVICE, SAVE_MODEL_PATH, early_stopper)
