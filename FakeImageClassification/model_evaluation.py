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