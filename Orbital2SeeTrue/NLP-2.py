import numpy as np
import pandas as pd
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast, pipeline
from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer
from transformers import AlbertTokenizer, AlbertModel

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

import torch
if torch.cuda.is_available():
    print("GPU is available")
else:
    print("GPU is NOT available")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Variable used below to set model training mode to use GPU

import os
# CUBLAS_WORKSPACE_CONFIG=:4096:8 needed for pytorch reproducibility
# https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260/3
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import random


def set_seed_all(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


set_seed_all(1)

import pandas as pd

DATASET_ROOT = '.'
DATASET_TRAIN_FOLDER = os.path.join(DATASET_ROOT, 'train')
DATASET_TRAIN_CSV = os.path.join(DATASET_ROOT, 'train.csv')
DATASET_TEST_FOLDER = os.path.join(DATASET_ROOT, 'test')
DATASET_TEST_CSV = os.path.join(DATASET_ROOT, 'test.csv')

train_df = pd.read_csv(DATASET_TRAIN_CSV)
# nlp_class_counts = train_df['nlp_class'].value_counts()
# print(nlp_class_counts)
# cv_class_counts = train_df['cv_class'].value_counts()
# print(cv_class_counts)


# Instantiate pipeline with fill-mask task
# unmasker = pipeline('fill-mask', model='albert-base-v2')
# other than bert you can try other pretrained models from huggingface e.g. albert, distilbert, etc

# But it can be biased e.g. Occupation for Male vs Female
# print(unmasker("Chloe wishes to be a [MASK]."))

# Named entity recognition
# ner = pipeline(model="dslim/bert-base-NER-uncased")

# MODEL = "jy46604790/Fake-News-Bert-Detect"
# clf = pipeline("text-classification", model=MODEL, tokenizer=MODEL)
# print(clf("Indonesian police have recaptured a U.S. citizen who escaped a week ago from an overcrowded prison on the"))



data = pd.read_csv('train.csv')
data_filled = data.fillna(value="")
# data['nlp_title_text'] = "title: " + data_filled['nlp_title'].astype(str) + ", text: " + data_filled['nlp_text'].astype(str)

data['nlp_title_text'] = data_filled['nlp_title'].astype(str) + " " + data_filled['nlp_comments'].astype(str)
print(data.nlp_title_text[26])

# Target column is made of string values True/Fake, let's change it to numbers 0/1 (Fake=1)
data['label'] = pd.get_dummies(data.nlp_class)['fake']


# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['nlp_title_text'], data['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=data['nlp_class'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)

# Load BERT model and tokenizer via HuggingFace Transformers https://huggingface.co/models
bert = AlbertModel.from_pretrained("albert-base-v2")
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

# bert = RobertaModel.from_pretrained("jy46604790/Fake-News-Bert-Detect")
# tokenizer = RobertaTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")

# bert = RobertaModel.from_pretrained("textattack/roberta-base-CoLA")
# tokenizer = RobertaTokenizer.from_pretrained("textattack/roberta-base-CoLA")

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert = BertModel.from_pretrained("bert-base-uncased")

# Plot histogram of the number of words in train data 'title'
seq_len = [len(title.split()) for title in train_text]

pd.Series(seq_len).hist(bins = 100,color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
# plt.show()


# Majority of titles above have word length under 15. So, we set max title length as 15
MAX_LENGTH = 250
# Tokenize and encode sequences in the train set
tokens_train = tokenizer.batch_encode_plus(
    train_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the validation set
tokens_val = tokenizer.batch_encode_plus(
    val_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)
# tokenize and encode sequences in the test set
tokens_test = tokenizer.batch_encode_plus(
    test_text.tolist(),
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

# Convert lists to tensors
train_seq = torch.tensor(tokens_train['input_ids'])
train_mask = torch.tensor(tokens_train['attention_mask'])
train_y = torch.tensor(train_labels.tolist())

val_seq = torch.tensor(tokens_val['input_ids'])
val_mask = torch.tensor(tokens_val['attention_mask'])
val_y = torch.tensor(val_labels.tolist())

test_seq = torch.tensor(tokens_test['input_ids']).to(device)
test_mask = torch.tensor(tokens_test['attention_mask']).to(device)
test_y = torch.tensor(test_labels.tolist()).to(device)

# Data Loader structure definition
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 64                                              #define a batch size

train_data = TensorDataset(train_seq, train_mask, train_y)    # wrap tensors
train_sampler = RandomSampler(train_data)                     # sampler for sampling the data during training
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                                                              # dataLoader for train set
val_data = TensorDataset(val_seq, val_mask, val_y)            # wrap tensors
val_sampler = SequentialSampler(val_data)                     # sampler for sampling the data during training
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
                                                              # dataLoader for validation set

# Freezing the parameters and defining trainable BERT structure
for param in bert.parameters():
    param.requires_grad = False    # false here means gradient need not be computed


class BERT_Arch(nn.Module):
    def __init__(self, bert):
      super(BERT_Arch, self).__init__()
      self.bert = bert
      self.dropout = nn.Dropout(0.1)            # dropout layer
      self.relu =  nn.ReLU()                    # relu activation function
      self.fc1 = nn.Linear(768,512)             # dense layer 1
      self.fc2 = nn.Linear(512,2)               # dense layer 2 (Output layer)
      self.softmax = nn.LogSoftmax(dim=1)       # softmax activation function
    def forward(self, sent_id, mask):           # define the forward pass
      cls_hs = self.bert(sent_id, attention_mask=mask)['pooler_output']
                                                # pass the inputs to the model
      x = self.fc1(cls_hs)
      x = self.relu(x)
      x = self.dropout(x)
      x = self.fc2(x)                           # output layer
      x = self.softmax(x)                       # apply softmax activation
      return x

model = BERT_Arch(bert)
model = model.cuda()
# Defining the hyperparameters (optimizer, weights of the classes and the epochs)
# Define the optimizer
from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = 5e-4)          # learning rate
# Define the loss function
cross_entropy  = nn.NLLLoss()
# Number of training epochs
epochs = 10


# Defining training and evaluation functions
def train():
  model.train()
  total_loss, total_accuracy = 0, 0

  for step,batch in enumerate(train_dataloader):                # iterate over batches
    if step % 50 == 0 and not step == 0:                        # progress update after every 50 batches.
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
    batch = [r.to(device) for r in batch]                                  # push the batch to gpu
    sent_id, mask, labels = batch
    model.zero_grad()                                           # clear previously calculated gradients
    preds = model(sent_id, mask)                                # get model predictions for current batch
    loss = cross_entropy(preds, labels)                         # compute loss between actual & predicted values
    total_loss = total_loss + loss.item()                       # add on to the total loss
    loss.backward()                                             # backward pass to calculate the gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)     # clip gradients to 1.0. It helps in preventing exploding gradient problem
    optimizer.step()                                            # update parameters
    preds=preds.detach().cpu().numpy()                          # model predictions are stored on GPU. So, push it to CPU

  avg_loss = total_loss / len(train_dataloader)                 # compute training loss of the epoch
                                                                # reshape predictions in form of (# samples, # classes)
  return avg_loss                                 # returns the loss and predictions

def evaluate():
  print("\nEvaluating...")
  model.eval()                                    # Deactivate dropout layers
  total_loss, total_accuracy = 0, 0
  for step,batch in enumerate(val_dataloader):    # Iterate over batches
    if step % 50 == 0 and not step == 0:          # Progress update every 50 batches.
                                                  # Calculate elapsed time in minutes.
                                                  # Elapsed = format_time(time.time() - t0)
      print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))
                                                  # Report progress
    batch = [t.to(device) for t in batch]                    # Push the batch to GPU
    sent_id, mask, labels = batch
    with torch.no_grad():                         # Deactivate autograd
      preds = model(sent_id, mask)                # Model predictions
      loss = cross_entropy(preds,labels)          # Compute the validation loss between actual and predicted values
      total_loss = total_loss + loss.item()
      preds = preds.detach().cpu().numpy()
  avg_loss = total_loss / len(val_dataloader)         # compute the validation loss of the epoch
  return avg_loss


# Train and predict (OPTIONAL)
best_valid_loss = float('inf')
train_losses=[]                   # empty lists to store training and validation loss of each epoch
valid_losses=[]

for epoch in range(epochs):
    print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
    train_loss = train()                       # train model
    valid_loss = evaluate()                    # evaluate model
    if valid_loss < best_valid_loss:              # save the best model
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'new_model_weights2.pt')
    train_losses.append(train_loss)               # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')


# save weights of trained model (OPTIONAL)
path = 'fakenews_weights2.pt'
torch.save(model.state_dict(), path)

with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y.cpu(), preds))

