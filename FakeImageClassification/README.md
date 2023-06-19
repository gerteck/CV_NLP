# Fake Image Classification
## Computer Vision and Machine Learning


Firstly, we need to verify if GPU is available to speed up model training.
If operating on pyCharm, we need to first install 
- Nvidia CUDA toolkit ( to run on supported local GPU )
- torch package
- numPy package

If GPU is unavailable through PyCharm due to the virtual environment, 
you can opt to run the py file through the command line.

```commandline
py .\fake_image_classifier.py
```
- Otherwise, set python interpreter as your system interpreter.

### Set up files, download and verify libraries as needed
Verify GPU Runtime is available, GPU Runtime is essential to speed up model training.
```python
import torch
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Variable used below to set model training mode to use GPU
```

Note: Data file (.zip) is already present in repository so this is not necessary.
```commandline
!pip3 list | grep torch
!wget -nc -O dataset.zip https://www.dropbox.com/s/zp70wf6a7cnqjih/CV_Workshop_Dataset.zip?dl=1
!unzip -nq dataset.zip
```
### To ensure model reproduciblity:
Training using common deep learning libraries (PyTorch, Tensorflow) with the same set of data and model hyper-parameters
would yield different results. They internally use different randomizers e.g. randomizer for train/val split or 
minibatch selection. Read [here](https://pytorch.org/docs/stable/notes/randomness.html).

It is strongly recommended to remove controllable sources of randomness for the purpose of verifying model 
performances by making the experiment reproducible, if the same code is ran again. We accomplish this by setting 
a fixed random seed. We should only initalize once, at start of code.

```python
import os
# CUBLAS_WORKSPACE_CONFIG=:4096:8 needed for pytorch reproducibility
# https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random
import torch
import numpy as np

def set_seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed_all(1)
```

## Dataset Statistics:
Get data statistics to understand quantity/quality of data. Helps you to plan what kind of data pre-processing is 
required.
```python
import pandas as pd
import os

# Pandas is a table library in python
# May want to consider which class has least images to determine how
# to train the model.

# Here, we extract the zip folder into its own folder.
DATASET_ROOT = './CV_Workshop_Dataset'
DATASET_TRAIN_FOLDER = os.path.join(DATASET_ROOT, 'train')
DATASET_TRAIN_CSV = os.path.join(DATASET_ROOT, 'train.csv')
DATASET_TEST_FOLDER = os.path.join(DATASET_ROOT, 'test')
DATASET_TEST_CSV = os.path.join(DATASET_ROOT, 'test.csv')

train_df = pd.read_csv(DATASET_TRAIN_CSV)
class_counts = train_df['class'].value_counts()
print(class_counts)
print(train_df)
```

Display a few samples of real and fake images:
```python
# Samples of images:
import climage

def PrintImageSamples(image_paths, label):
    for i, path in enumerate(image_paths):
        print(str(i) + ": " + str(label))
        output = climage.convert(path)
        print(output)

SAMPLES = 8
real_image_paths = train_df[train_df['class'] == 'real'][:SAMPLES]['filename']
real_image_paths = [os.path.join(DATASET_ROOT, p) for p in real_image_paths]
PrintImageSamples(real_image_paths, 'real')

fake_image_paths = train_df[train_df['class'] == 'fake'][:SAMPLES]['filename']
fake_image_paths = [os.path.join(DATASET_ROOT, p) for p in fake_image_paths]
PrintImageSamples(fake_image_paths, 'fake')
```


## Data Pre-Processing
### Pytorch custom dataset
In Pytorch, the Dataset and Dataloader class is used to customize your dataset files into a python iterator which 
can be called to retrieve batches of image & label samples for training.

- Creating Pytorch Dataset [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files)
- Pytorch Dataloader using Dataset [here](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders)

### Train & Validation split
Portion out some training images for validation. Model needs to use validation images to provide unbiased assessment 
of its performance during training.

### Image Pre-processing & Augmentation (Pytorch Transforms)
- Writing a Pytorch transform compose [here](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html#afterword-torchvision)
- Available torchvision transforms [here](https://pytorch.org/vision/stable/transforms.html)
```python
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
```

Transforming training set as well as portioning train image for validation:

```python
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
```
```python
# Portion out some train images for validation
TRAIN_VALID_SPLIT = [0.8, 0.2] # Ratio of train/validation split

labelled_dataset = ImageDatasetFromCSV(annotations_csv=DATASET_TRAIN_CSV, root_dir=DATASET_ROOT, has_labels=True, transform=train_transform)
train_dataset, valid_dataset = random_split(labelled_dataset, TRAIN_VALID_SPLIT)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# Create dataset from testing unlabelled data. Set has_labels = False since no label in csv.
test_dataset = ImageDatasetFromCSV(annotations_csv=DATASET_TEST_CSV, root_dir=DATASET_ROOT, has_labels=False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Dataset statistics
print(f"Total Samples - Train:{len(train_dataset)}, Val:{len(valid_dataset)}, Test:{len(test_dataset)}")
```
- Note: If a RunTimeError: CUDA out of memory is encountered, one may attempt to halve the batch_size. 

### Next, we need to train and test dataloader. 
- A dataloader provides an efficient and flexible way to load data into a model for training / inference.
- See [link](https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#iterate-through-the-dataloader)

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from glob import glob

# Train dataloader
extracted_batch = next(iter(train_dataloader)) # Get first batch
print(f"Images batch shape: {extracted_batch['image'].size()}")
print(f"Labels batch shape: {extracted_batch['label'].size()}")
extracted_image = extracted_batch['image'][0].squeeze() # Take 1 image to visualize
extracted_label_idx = extracted_batch['label'][0] # Take corresponding label index of image
label = list(LABELS_MAP.keys())[list(LABELS_MAP.values()).index(extracted_label_idx)] # Convert label index back to text
print(f"Label Index: {extracted_label_idx}, Label Class: {label}")

# rcParams['figure.figsize'] = 4,4 # Change display figure size
plt.imshow(np.transpose(extracted_image, (1, 2, 0))) # Convert pytorch C H W to numpy image H W C format
plt.show()

# Test dataloader
extracted_test_batch = next(iter(test_dataloader))
print(f"Images batch shape: {extracted_test_batch['image'].size()}")
extracted_image = extracted_test_batch['image'][0].squeeze()
plt.imshow(np.transpose(extracted_image, (1, 2, 0)))
plt.show()
```

## Model Selection & Configuration
Pytorch provides SOTA classification models from their torchvision [library](https://pytorch.org/vision/stable/models.html#classification) 
or [torch.hub](https://pytorch.org/vision/stable/models.html#using-models-from-hub)

The example below uses the resnet50 model with pre-trained weight (model input shape is 224x224)

NOTE: Check a pre-trained model's fully connected (fc) layer output dimensions, modify model's fc layer to 
match # of classes in dataset.
```python
# For example, notice the dimensions for Resnet-50
print(model)
...
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=2048, out_features=1000, bias=True)
)
```
```python
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
```
### Hyper-parameter Tuning
Pytorch Optimizers [here](https://pytorch.org/docs/stable/optim.html)

[CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss)
[BCEWithLogitsLoss](https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html#torch.nn.BCEWithLogitsLoss)
https://discuss.pytorch.org/t/softmax-cross-entropy-loss/125383

```python
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Adadelta, RMSprop # Optimizers

criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(),lr=0.001, momentum=0.9) # Configure learning rate parameters as needed
```

### Early Stopping
Create your own early stopping function. [Example](https://stackoverflow.com/a/73704579)

```python
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
```

### Model Training

Implement a training loop example 
[here](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#train-the-network)

```python
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

```
```python
EPOCH = 3
SAVE_MODEL_PATH = 'model.pt'
train(model, train_dataloader, valid_dataloader, optimizer, criterion, EPOCH, DEVICE, SAVE_MODEL_PATH, early_stopper)
```

Sample Output from terminal:
```commandline
Epoch 0, Train loss: 0.59559: 100%|██████████| 40/40 [00:12<00:00,  3.31it/s]
Epoch 0, Valid loss: 0.49241: 100%|██████████| 10/10 [00:01<00:00,  9.06it/s]
Summary: Epoch 0 - Train loss: 0.59559, Valid loss: 0.49241
Epoch 1, Train loss: 0.32794: 100%|██████████| 40/40 [00:09<00:00,  4.16it/s]
Epoch 1, Valid loss: 0.29565: 100%|██████████| 10/10 [00:01<00:00,  9.18it/s]
Summary: Epoch 1 - Train loss: 0.32794, Valid loss: 0.29565
Epoch 2, Train loss: 0.14578: 100%|██████████| 40/40 [00:10<00:00,  3.91it/s]
Epoch 2, Valid loss: 0.15811: 100%|██████████| 10/10 [00:01<00:00,  8.74it/s]
Summary: Epoch 2 - Train loss: 0.14578, Valid loss: 0.15811
```

## Model Evaluation

After best model has been selected based on the validation score, run test images through model to get predictions.
We then save it into a csv (for submission and verification of results).

- Refer to model_evaluation.py for python script.

```python
import os
import torch
from torch.nn import Linear
from torchvision.models import resnet50, ResNet50_Weights
import pandas as pd

# Setting up GPU for model training.
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Variable used below to set model training mode to use GPU

LABELS_MAP = { # Convert text labels to class index, model is trained using class indexes
    'real': 0 ,
    'fake': 1 }
NUM_CLASS = len(LABELS_MAP)
def MyModel():
    pre_trained = ResNet50_Weights.DEFAULT # Choose to use pre-trained weights or not (train from scratch)
    model = resnet50(weights=pre_trained)
    model.fc = Linear(in_features=2048, out_features = NUM_CLASS) # Change out last fully connected layer to match number of classes in dataset
    return model


# # Load model from downloaded checkpoint file
model = MyModel()
checkpoint = torch.load('model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Evaluation
from tqdm import tqdm
from time import sleep
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.io import read_image # Load image file path to ndarray

# Dataset Statistics
DATASET_ROOT = './CV_Workshop_Dataset'
DATASET_TEST_CSV = os.path.join(DATASET_ROOT, 'test.csv')

# Test set Transforms
TEST_IM_W = 244
TEST_IM_H = 244
test_transform = transforms.Compose([
        transforms.Resize((TEST_IM_W, TEST_IM_H)),
        transforms.ConvertImageDtype(dtype=torch.float32)])


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


test_dataset = ImageDatasetFromCSV(annotations_csv=DATASET_TEST_CSV, root_dir=DATASET_ROOT, has_labels=False, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

def evaluate(model, test_dataloader, device):
    with torch.no_grad(): # https://discuss.pytorch.org/t/model-eval-vs-with-torch-no-grad/19615/2
        model.eval() # Affects batch normalization and dropout layers (they work differently in training vs evaluation mode)
        model.to(device)
        predictions = []
        test_set = tqdm(test_dataloader)
        for batch_idx, batch in enumerate(test_set):
            images = batch['image']
            images = images.to(device)
            prediction = model(images)
            # If SoftmaxCrossEntopy loss was used, output is logits, convert to probability scores
            prediction = torch.nn.functional.softmax(prediction, dim=1)
            predictions.extend(prediction.tolist())
        return predictions

predictions = evaluate(model, test_dataloader, DEVICE)
print(predictions)

# if output of model was 2 class ( multi-class ) probability scores, take the first value to convert to prediction scores for binary class
predictions = [x[1] for x in predictions]
print(predictions)

# Write to csv predictions
test_df = pd.read_csv(DATASET_TEST_CSV)
test_df['class'] = predictions
test_df.to_csv('predictions.csv', index=False)

# Recall:
# LABELS_MAP = { # Convert text labels to class index, model is trained using class indexes
#     'real': 0 ,
#     'fake': 1 }
# NUM_CLASS = len(LABELS_MAP)
```

## End
Credit and reference materials from DSTA Hackathon Workshop Organisers (Official Open Materials). 
Adapted to operate locally on PyCharm.

### All Imports:
```python
import os
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from torchvision.io import read_image # Load image file path to ndarray
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
from glob import glob
from torch.nn import Linear
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam, Adadelta, RMSprop # Optimizers
from tqdm import tqdm
from time import sleep
```