# Fake News Predictions
## Natural Language Processing and Machine Learning

## Setup Environment

```commandline
REM ## Install specific libraries

pip install transformers
pip install pycaret
pip install lime
```
Note:
- pandas (pycaret) is not enabled for python 3.11 (as of now), hence we can run following to install for py 3.9.
- Following that, we will need to run on python 3.9. Hence, set the system interpreter accordingly.

```commandline
py -3.9 -m pip install pycaret
```


```python
import numpy as np
import pandas as pd
import pycaret
import transformers
from transformers import AutoModel, BertTokenizerFast, pipeline
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
# specify GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.')
    print('Device name:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")
```

### Introduction to Pretrained Models e.g. BERT

A pre-trained model is a saved network that was previously trained on a large dataset. 
Pretrained models use transfer learning to customize itself to a given task e.g. NER, sentiment analysis, etc. 
They have the benefit of not needing as much data as building a model from scratch. Other than BERT there are 
other pretrained models or even fine-tuned models you can get from https://huggingface.co/models

```python
# Instantiate pipeline with fill-mask task
unmasker = pipeline('fill-mask', model='bert-base-uncased')
# other than bert you can try other pretrained models from huggingface e.g. albert, distilbert, etc

# The pretrined model knows abit of Geography e.g. Ang Mo Kio vs Johor
unmasker("ang mo kio is located in [MASK].")

# Gender e.g. xiao mei vs da xiong
unmasker("guo xiong just arrived. [MASK] is over there.")

# and even Ethnicity e.g. da xiong vs muthusamy
#unmasker("muthusamy's ethnicity is definitely [MASK].")

# But it can be biased e.g. Occupation for Male vs Female
unmasker("Jenny is working as a [MASK].")

# or even outright wrong
unmasker("Singapore was founded on 9 August [MASK].")
```

dslim/bert-base-NER-uncased is an example of pretrained uncased bert finetuned to do NER:

```python
ner = pipeline(model="dslim/bert-base-NER-uncased")
# apple can be extracted as an organisation or a person or place depending on how it is being used in a sentence
# apple makes iphones
# my name is apple
# i stay in apple
# i just ate an apple
ner("my name is apple")

```

## Load Dataset
```python
# Load Dataset
true_data = pd.read_csv('true.csv')
fake_data = pd.read_csv('fake.csv')

# Generate labels True/Fake under new Target Column in 'true_data' and 'fake_data'
true_data['Target'] = ['True']*len(true_data)
fake_data['Target'] = ['Fake']*len(fake_data)

# Merge 'true_data' and 'fake_data', by random mixing into a single df called 'data'
data = true_data.append(fake_data).sample(frac=1).reset_index().drop(columns=['index'])

# See how the data looks like
print(data.shape)
data.head()

# Target column is made of string values True/Fake, let's change it to numbers 0/1 (Fake=1)
data['label'] = pd.get_dummies(data.Target)['Fake']

# Checking if our data is well balanced
label_size = [data['label'].sum(),len(data['label'])-data['label'].sum()]
plt.pie(label_size,explode=[0.1,0.1],colors=['firebrick','navy'],startangle=90,shadow=True,labels=['Fake','True'],autopct='%1.1f%%')
```

### Train Test Split
Train-Validation-Test set split into 70:15:15 ratio
```python
# Train-Validation-Test set split into 70:15:15 ratio
# Train-Temp split
train_text, temp_text, train_labels, temp_labels = train_test_split(data['title'], data['label'],
                                                                    random_state=2018,
                                                                    test_size=0.3,
                                                                    stratify=data['Target'])
# Validation-Test split
val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels,
                                                                random_state=2018,
                                                                test_size=0.5,
                                                                stratify=temp_labels)
```
## BERT Fine-tuning

## Load pretrained BERT Model and Tokenizer

```python
# Load BERT model and tokenizer via HuggingFace Transformers https://huggingface.co/models
bert = AutoModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
```

```python
# BERT Tokenizer Functionality
# 1. Start token is 101, End token is 102
# 2. Certain words are break up into sub words e.g. subword is broken into sub ##word
samples = ["snowboard snowball snowman.",
           "subclass subrent."]

for sample in samples:
  tokens = tokenizer.tokenize(sample)
  print("tokens:      ", tokens)
  tokenized_sample = tokenizer.batch_encode_plus(
      [sample],
      max_length = 10,
      pad_to_max_length=True,
      truncation=True
  )
  print(tokenized_sample)

# Ref: https://huggingface.co/docs/transformers/preprocessing
```

### Prepare Input Data
```python
# Plot histogram of the number of words in train data 'title'
seq_len = [len(title.split()) for title in train_text]

pd.Series(seq_len).hist(bins = 40,color='firebrick')
plt.xlabel('Number of Words')
plt.ylabel('Number of texts')
```

```python
# Majority of titles above have word length under 15. So, we set max title length as 15
MAX_LENGTH = 15
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
```

```python
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
```
```python
# Data Loader structure definition
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
batch_size = 32                                               #define a batch size

train_data = TensorDataset(train_seq, train_mask, train_y)    # wrap tensors
train_sampler = RandomSampler(train_data)                     # sampler for sampling the data during training
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
                                                              # dataLoader for train set
val_data = TensorDataset(val_seq, val_mask, val_y)            # wrap tensors
val_sampler = SequentialSampler(val_data)                     # sampler for sampling the data during training
val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)
                                                              # dataLoader for validation set
```

### Freeze Layers
```python
# Freezing the parameters and defining trainable BERT structure
for param in bert.parameters():
    param.requires_grad = False    # false here means gradient need not be computed
```

## Define Model Architecture and Hyperparameters
A hyperparameter is a ML parameter chosen before training starts. It should not be confused with parameters which refers 
to variables whose values are learnt during training.

The different hyperparameters to configure:

Optimizer: Adam

Learning Rate e.g. 1e-5 or 0.00001, Generally, a large learning rate allows the model to learn faster, at the cost of 
arriving on a sub-optimal final set of weights. A smaller learning rate may allow the model to learn a more optimal or 
even globally optimal set of weights but may take significantly longer to train.

Loss Function: CrossEntropyLoss. NLLLoss and CrossEntropy Loss are closely related. CrossEntropyLoss applies LogSoftmax 
to the output before passing it to NLLLoss.

No. of epochs: 2

In machine learning, a loss function and an optimizer are two essential components that help to improve the performance 
of a model. A loss function measures the difference between the predicted output of a model and the actual output, 
while an optimizer adjusts the model’s parameters to minimize the loss function.

```python
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
                  lr = 1e-5)          # learning rate
# Define the loss function
cross_entropy  = nn.NLLLoss()
# Number of training epochs
epochs = 2
```


### Define and Train & Evaluate Function

```python
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
```

### Model Training
Save the best model to new_model_weights.pt

```python
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
        torch.save(model.state_dict(), 'new_model_weights.pt')
    train_losses.append(train_loss)               # append training and validation loss
    valid_losses.append(valid_loss)

    print(f'\nTraining Loss: {train_loss:.3f}')
    print(f'Validation Loss: {valid_loss:.3f}')
```

```python
# save weights of trained model (OPTIONAL)
# path = 'fakenews_weights.pt'
# torch.save(model.state_dict(), path)
```

### Model Performance
```python
# load weights of best model
# path = 'fakenews_weights.pt'
# path = 'new_model_weights.pt'
# model.load_state_dict(torch.load(path))
```

To test on test split:

```python
with torch.no_grad():
  preds = model(test_seq, test_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
print(classification_report(test_y.cpu(), preds))
```

## Fake News Predictions

```python
# testing on unseen data (0 is True; 1 is Fake)

# Samples from Straits Times
#unseen_news_text = ["Singapore core and headline inflation ease in march, providing some respite"]
#unseen_news_text = ["Integrated resorts in Japan, Thailand set to compete with Singapore's own"]
unseen_news_text = ["Credit card spending on the rise in S'pore, but debts remain manageable"]

# Samples from Fake Websites
unseen_news_text = ["Depressed citizen dresses up as adult who has crap together"] # Curse Words
#unseen_news_text = ["Here’s The Moment A Black Woman Protected A White Man At A KKK Rally."]
#unseen_news_text = ["top snake handler leaves sinking huckabee campaign"]

# tokenize and encode sequences in the test set
MAX_LENGTH = 15
tokens_unseen = tokenizer.batch_encode_plus(
    unseen_news_text,
    max_length = MAX_LENGTH,
    pad_to_max_length=True,
    truncation=True
)

unseen_seq = torch.tensor(tokens_unseen['input_ids']).to(device)
unseen_mask = torch.tensor(tokens_unseen['attention_mask']).to(device)

with torch.no_grad():
  preds = model(unseen_seq, unseen_mask)
  preds = preds.detach().cpu().numpy()

preds = np.argmax(preds, axis = 1)
preds
```

## Explain Predictions with Lime
```python
def predict_proba(texts):
    # tokenize and encode sequences in the test set
    MAX_LENGTH = 15
    tokens_unseen = tokenizer.batch_encode_plus(
      texts,
      max_length = MAX_LENGTH,
      pad_to_max_length=True,
      truncation=True
    )

    unseen_seq = torch.tensor(tokens_unseen['input_ids']).to(device)
    unseen_mask = torch.tensor(tokens_unseen['attention_mask']).to(device)

    # with torch.no_grad():
    #   preds = model(unseen_seq, unseen_mask)
    # tensor_logits = preds
    # probas = F.softmax(tensor_logits).detach().numpy()
    # return probas

    with torch.no_grad():
      preds = model(unseen_seq, unseen_mask)
      preds = preds.detach().cpu().numpy()
    preds = np.exp(preds)
    return preds

predict_proba(unseen_news_text)
```

```python
class_names = ["True","Fake"]
explainer = LimeTextExplainer(class_names=class_names)
exp = explainer.explain_instance(unseen_news_text[0], predict_proba, num_features=10, num_samples=100)
exp.show_in_notebook(text=True)
```


Credit and reference materials from DSTA Hackathon Workshop Organisers (Official Open Materials). 
Adapted to operate locally on PyCharm.