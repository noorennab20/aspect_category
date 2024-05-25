#import libraries
import datetime
import os
import re
import random
import time

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
from torch.optim import SGD
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from transformers import BertConfig, BertModel

#Libraries for Pre-trained model from hugging face
from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


#clean the text 
#nltk.download("stopwords")
from nltk.corpus import stopwords


if torch.cuda.is_available():       
    device = torch.device("cuda")

else:
    device = torch.device("cpu")



def preprocess_arabic_tweets(df, column):
    # Remove URLs
    df[column] = df[column].apply(lambda x: re.sub(r'http\S+', '', x))
    
    # Remove mentions and hashtags
    df[column] = df[column].apply(lambda x: re.sub(r'@\w+|#\w+', '', x))
    
    # Remove punctuation and numbers
    df[column] = df[column].apply(lambda x: re.sub(r'[^\w\s]', '', x))
    df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x))
    
    # Remove stopwords
    arabic_stopwords = set(stopwords.words('arabic'))
    df[column] = df[column].apply(lambda x: ' '.join([word for word in x.split() if word not in arabic_stopwords]))
    
    return df


# Create a function to tokenize a set of texts
def preprocessing_for_bert(sentences):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []
    token_type_ids = []

    # For every sentence...
    for sentence in sentences:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            sentence,
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,                  # Max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_tensors='pt',           # Return PyTorch tensor
            return_attention_mask=True,
            truncation=True
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent['input_ids'])
        attention_masks.append(encoded_sent['attention_mask'])
        token_type_ids.append(encoded_sent['token_type_ids'])

    # Convert lists to tensors
    #input_ids = torch.tensor(input_ids)
    #attention_masks = torch.tensor(attention_masks)
    #token_type_ids=torch.tensor(token_type_ids)
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids=torch.cat(token_type_ids,dim=0)
    return input_ids, attention_masks,token_type_ids


class BertForAspectCategoryPolarity(nn.Module):
    def __init__(self, num_aspect_categories, num_polarities, dropout_prob=0.1):
        super().__init__()
        config = BertConfig.from_pretrained('asafaya/bert-base-arabic')  # Increase number of layers
        config.hidden_dropout_prob = dropout_prob  # Add dropout
        
        self.bert = BertModel(config=config)
        #self.aspect_category_classifier = nn.Linear(self.bert.config.hidden_size,num_aspect_categories)
        #self.polarity_classifier = nn.Linear(self.bert.config.hidden_size,num_polarities)
        self.aspect_category_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_aspect_categories)
            )
        self.polarity_classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_polarities)
            )
        self.aspect_category_weights = torch.tensor([1.2, 1.0, 1.5, 2.5,  2.2,1.5,1.0,1.2], dtype=torch.float32).to(device)
        self.polarity_weights = torch.tensor([0.1, 2.5], dtype=torch.float32).to(device)
    def forward(self,input_ids,attention_mask,aspect_category_labels=None,polarity_labels=None):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True)
        
        pooled_output = outputs.pooler_output
        aspect_category_logits = self.aspect_category_classifier(pooled_output)
        #print("aspect _logists",aspect_category_logits.shape)
        polarity_logits = self.polarity_classifier(pooled_output)
        #print("---------------------------------------------")
        #print("polarity_logits _logists",polarity_logits.shape)

        loss = None
        if aspect_category_labels is not None:
            aspect_category_labels = aspect_category_labels.to(torch.int64)
            #print("aspect_category_labels",aspect_category_labels.shape)
            num_aspect_categories = aspect_category_logits.shape[1]
            #print("num_aspect_categories",num_aspect_categories)
            aspect_category_labels_one_hot = F.one_hot(aspect_category_labels, num_classes=num_aspect_categories).float()
            #print("---------------------------------------------")
            #print("aspect_category_labels_one_hot",aspect_category_labels_one_hot.shape)
            
            
            #aspect_category_weights= torch.tensor([0.6, 0.5, 0.1, 0.3, 0.3, 0.3, 0.1])
            
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.aspect_category_weights)
            
            aspect_category_loss = loss_fct(aspect_category_logits.to(device), aspect_category_labels_one_hot.to(device))#.view(-1)
                        
            if polarity_labels is not None:
                polarity_labels = polarity_labels.to(torch.int64)
                #polarity_weights = torch.tensor([0.1, 0.6, 0.3])
                
                polarity_loss_fct = nn.CrossEntropyLoss(weight=self.polarity_weights)
                polarity_loss = polarity_loss_fct(polarity_logits.view(-1, num_polarities), polarity_labels.view(-1))
                
                loss = aspect_category_loss + polarity_loss
            else:
                loss = aspect_category_loss

        return loss, aspect_category_logits, polarity_logits




#class of evaluation
def evaluate(model, dataloader):
    model.eval()
    y_true_aspect_category = []
    y_pred_aspect_category = []
    y_true_polarity = []
    y_pred_polarity = []
    total_loss = 0
    total_aspect_category_acc = 0
    total_polarity_acc = 0
    total_loss = 0
    

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, aspect_category_labels,polarity_labels = batch
            input_ids=input_ids.to(device)
            attention_mask=attention_mask.to(device)
            aspect_category_labels=aspect_category_labels.to(device)
            polarity_labels=polarity_labels.to(device)
            
            _,aspect_category_logits, polarity_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # One-hot encode the aspect_category_labels tensor
            aspect_category_labels = aspect_category_labels.to(torch.int64)
            num_aspect_categories = aspect_category_logits.shape[-1]

            aspect_category_labels_onehot = F.one_hot(aspect_category_labels, num_classes=num_aspect_categories).float()
            
            # Forward pass
            
            aspect_category_loss = F.binary_cross_entropy_with_logits(aspect_category_logits, aspect_category_labels_onehot)
            polarity_loss = F.cross_entropy(polarity_logits, polarity_labels.long())

            # Compute total loss and accuracy
            loss = aspect_category_loss + polarity_loss
            total_loss += loss.item()
            
            # Compute accuracy
            aspect_category_preds = torch.round(torch.sigmoid(aspect_category_logits))
            aspect_category_acc = (aspect_category_preds == aspect_category_labels_onehot).all(dim=1).float().mean().item()
            polarity_preds = torch.argmax(polarity_logits, dim=1)
            polarity_acc = (polarity_preds == polarity_labels).float().mean().item()

            total_aspect_category_acc += aspect_category_acc
            total_polarity_acc += polarity_acc
    avg_aspect_category_acc = total_aspect_category_acc / len(dataloader)
    avg_polarity_acc = total_polarity_acc / len(dataloader)
    avg_loss = total_loss / len(dataloader)
    # Check for improvement in validation loss
    return avg_aspect_category_acc, avg_polarity_acc, avg_loss


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))



 # Specify loss function
loss_fn = nn.CrossEntropyLoss()

 # Load data and set labels
data_complaint = pd.read_excel('dataset/POS.xlsx')
data_complaint['polarity'] = 0
data_non_complaint = pd.read_excel('dataset/NEG.xlsx')
data_non_complaint['polarity'] = 1
# Concatenate complaining and non-complaining data
data = pd.concat([data_complaint, data_non_complaint], axis=0).reset_index(drop=True)
sampleTrain= data.sample(5)
#print(sampleTrain)

data = data.rename(columns={'polarity': 'label'})
data = data.rename(columns={'opinion_category': 'category'})


clean_data=preprocess_arabic_tweets(data,"text")
clean_data=clean_data.copy()

# Unique category
fig=plt.figure(figsize=(5,4))
clean_data['label'].hist()

# Unique category
fig=plt.figure(figsize=(110,20))
clean_data['category'].hist(bins='fd')

# Deleting the column because it is so less
clean_data.drop(clean_data[clean_data['category'] == 'SOUNDS#QUALITY'].index, inplace=True)

# Deleting the neutal column because it is so less
clean_data.drop(clean_data[clean_data['category'] == 'PERFORMANCE#POWER CONSUMPTION'].index, inplace=True)

clean_data.replace('PLAYABILITY#INTERNET CONNECTION','PLAYABILITY', inplace=True)
clean_data.replace('PERFORMANCE#INTERNET CONNECTION','PERFORMANCE', inplace=True)
clean_data.replace('PERFORMANCE#RESPONSE TIME','PERFORMANCE', inplace=True)
clean_data.replace('GAMEPLAY#ENJOYMENT','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#CHALLENGE','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#GENERAL','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#LEVELS','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#SOCIALIZATION','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#FAIRNESS','GAMEPLAY', inplace=True)
clean_data.replace('GAMEPLAY#STORY','GAMEPLAY', inplace=True)
clean_data.replace('PLAYABILITY#PRICE','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#LOCALIZATION','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#ADS','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#CONTROLS','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#LEARNABILITY','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#COMPATIBILITY','PLAYABILITY', inplace=True)
clean_data.replace('PLAYABILITY#GENERAL','PLAYABILITY', inplace=True)
clean_data.replace('PERFORMANCE#STABILITY','PERFORMANCE', inplace=True)
clean_data.replace('PERFORMANCE#GENERAL','PERFORMANCE', inplace=True)
clean_data.replace('GRAPHICS#ART','GRAPHICS', inplace=True)
clean_data.replace('GRAPHICS#GENERAL','GRAPHICS', inplace=True)
clean_data.replace('GRAPHICS#QUALITY','GRAPHICS', inplace=True)
clean_data.replace('UPDATES#GENERAL','UPDATES', inplace=True)
clean_data.replace('SOUNDS#GENERAL','SOUNDS', inplace=True)
clean_data.replace('SECURITY#GENERAL','SECURITY', inplace=True)
clean_data.replace('APP#GENERAL','app', inplace=True)
clean_data['category'].value_counts()


# Unique category
fig=plt.figure(figsize=(10,10))
clean_data['category'].hist(bins='fd')


#Spilting the data set into train , validation and test dataset
df_train, df_val, df_test = np.split(clean_data.sample(frac=1, random_state=27),
                            [int(.6 * len(clean_data)), int(.7 * len(clean_data))])



# Create label encoders for category and polarity
category_encoder = LabelEncoder()
polarity_encoder = LabelEncoder()

# Fit the encoders to the train data
category_encoder.fit(df_train['category'])
polarity_encoder.fit(df_train['label'])

# Transform the category and label columns for train, val, and test dataframes
df_train['category'] = category_encoder.transform(df_train['category'])
df_train['label'] = polarity_encoder.transform(df_train['label'])

df_val['category'] = category_encoder.transform(df_val['category'])
df_val['label'] = polarity_encoder.transform(df_val['label'])

df_test['category'] = category_encoder.transform(df_test['category'])
df_test['label'] = polarity_encoder.transform(df_test['label'])


# Print the class labels for each target variable
print(polarity_encoder.classes_)  # ['negative' 'neutral' 'positive']
print(category_encoder.classes_)  

#Tokenization of the sentences
MAX_LEN=64

# Load the BERT tokenizer
print('Tokenizing data...')
tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')#use_fast=False
## Train
train_inputs, train_masks,token_type_ids = preprocessing_for_bert(df_train['text'])
df_train = df_train.reset_index(drop=True)
train_category = torch.tensor(df_train.category)
train_label = torch.tensor(df_train.label)
## Validation
val_inputs, val_masks,val_type_ids = preprocessing_for_bert(df_val['text'])
df_val = df_val.reset_index(drop=True)
validation_category = torch.tensor(df_val['category'])
validation_label = torch.tensor(df_val['label'])
## Test data
test_inputs, test_masks,test_type_ids = preprocessing_for_bert(df_test['text'])
df_test = df_test.reset_index(drop=True)
test_category = torch.tensor(df_test['category'])
test_label = torch.tensor(df_test['label'])
print("Done !")


#Data Loaders for Label
batch_size = 24
# Create the DataLoader for our training set
train_data = TensorDataset(train_inputs, train_masks, train_category, train_label)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
# Create the DataLoader for our validation set
val_data = TensorDataset(val_inputs, val_masks, validation_category,validation_label)
val_sampler = SequentialSampler(val_data)
val_dataloader = DataLoader(val_data,sampler=val_sampler,batch_size=batch_size)
# Create the DataLoader for our validation set
test_data = TensorDataset(test_inputs, test_masks, test_category,test_label)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data,sampler=test_sampler,batch_size=batch_size)



# Assuming that your aspect category labels are stored as integers in a 1D tensor
num_aspect_categories = int(train_category.max() + 1)
num_aspect_categories

num_polarities = int(train_label.max() + 1)
num_polarities
clean_data['category'].value_counts()


#Create BertClassifier
hidden_size=500

#Optimizer & Learning Rate Scheduler    
learning_rate=2e-5
num_epochs=10
# Initialize variables for early stopping
early_stopping_counter = 0
patience = 5
best_val_loss = float('inf')
accumulation_steps = 4

num_aspect_categories


model = BertForAspectCategoryPolarity(num_aspect_categories=num_aspect_categories, num_polarities=num_polarities)



#Training the model
evaluation=True
model = BertForAspectCategoryPolarity(num_aspect_categories=num_aspect_categories, num_polarities=num_polarities)


optimizer = AdamW(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#optimizer = optim.Adagrad(model.parameters(), lr=0.01)


# Initialize lists to store training loss and epoch number
train_loss_list = []
epoch_list = []
val_loss_list=[]
cat_acc_list=[]
pol_acc_list=[]

best_acc = 0
best_loss = 1000
best_val_loss = float('inf')
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)
# Fine-tune the model on the training data for aspect category detection
for epoch in range(num_epochs):
    # =======================================
    #               Training
    # =======================================
    # Print the header of the result table
    print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Catergory Acc':^9} | {'Val polarity Acc':^9} | {'Elapsed':^9}")
    print("-"*80)
    # Measure the elapsed time of each epoch
    t0_epoch, t0_batch = time.time(), time.time()

    # Reset tracking variables at the beginning of each epoch
    total_loss, batch_loss, batch_counts = 0, 0, 0
    
    # Set the model to training mode
    model.train()
    
    # Reset the total loss for this epoch.
    total_train_loss = 0
    
    # Measure how long the training epoch takes.
    t0 = time.time()
    
    # Iterate over batches of training data
    for step, batch in enumerate(tqdm(train_dataloader)): #, desc="Training"
        batch_counts +=1
        # Unpack batch data
        input_ids, attention_mask, aspect_category_labels,polarity_labels = batch
        input_ids=input_ids.to(device)
        attention_mask=attention_mask.to(device)
        aspect_category_labels=aspect_category_labels.to(device)
        polarity_labels=polarity_labels.to(device)
        
        # Zero out any previously calculated gradients
        optimizer.zero_grad()
        
       
        loss, aspect_category_logits, polarity_logits= model(input_ids=input_ids,
                                                             attention_mask=attention_mask,
                                                             aspect_category_labels=aspect_category_labels,
                                                             polarity_labels=polarity_labels)
        
        # Calculate loss
        
        batch_loss += loss.item()
        
        total_train_loss += loss.item()
        
        # Perform a backward pass to calculate gradients 
        loss.backward() 
        # Update weights based on calculated gradients 
        optimizer.step() 
        
        # Print the loss values and time elapsed for every 20 batches
        """if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
            # Calculate time elapsed for 20 batches
            time_elapsed = time.time() - t0_batch

            # Print training results
            print(f"{epoch + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

            # Reset batch tracking variables
            batch_loss, batch_counts = 0, 0
            t0_batch = time.time()"""
     # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)
    train_loss_list.append(avg_train_loss)
    epoch_list.append(epoch+1)
    
    
    print("-"*80)
    # =======================================
    #               Evaluation
    # =======================================
    if evaluation == True:
        # After the completion of each training epoch, measure the model's performance
        # on our validation set.
        aspect_category_acc, polarity_acc, avg_loss = evaluate(model, val_dataloader)
        val_loss_list.append(avg_loss)
        cat_acc_list.append(aspect_category_acc)
        pol_acc_list.append(polarity_acc)

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        
        print("---------------------- Evaluate the model -------------------")
            
        print(f"{epoch + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {avg_loss:^10.6f} | {aspect_category_acc:^9.3f} | {polarity_acc:^9.3f} | {time_elapsed:^9.2f}")
        print("-"*70)
        print("\n")
        
        
    
    print("Training complete!")



#Test the model
def test1(model, dataloader):
    model.eval()
    device = next(model.parameters()).device
    predictions = []
    aspect_category_true = []
    polarity_true = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, aspect_category_labels, polarity_labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            aspect_category_labels = aspect_category_labels.to(device)
            polarity_labels = polarity_labels.to(device)
            _, aspect_category_logits, polarity_logits = model(input_ids, attention_mask)
            aspect_category_predictions = aspect_category_logits.argmax(dim=1).cpu().tolist()
            polarity_predictions = polarity_logits.argmax(dim=1).cpu().tolist()
            predictions.extend(list(zip(aspect_category_predictions, polarity_predictions)))
            aspect_category_true.extend(aspect_category_labels.cpu().tolist())
            polarity_true.extend(polarity_labels.cpu().tolist())

    aspect_category_accuracy = accuracy_score(aspect_category_true, [x[0] for x in predictions])
    polarity_accuracy = accuracy_score(polarity_true, [x[1] for x in predictions])
    aspect_category_f1 = f1_score(aspect_category_true, [x[0] for x in predictions], average='weighted')
    polarity_f1 = f1_score(polarity_true, [x[1] for x in predictions], average='weighted')
    
    
    micro_aspect_category_f1 = f1_score(aspect_category_true, [x[0] for x in predictions], average='micro')
    micro_polarity_f1 = f1_score(polarity_true, [x[1] for x in predictions], average='micro')
    
    mac_aspect_category_f1 = f1_score(aspect_category_true, [x[0] for x in predictions], average='macro')
    mac_polarity_f1 = f1_score(polarity_true, [x[1] for x in predictions], average='macro')
    
    print(f"Aspect category accuracy: {aspect_category_accuracy:.4f}")
    print(f"Polarity accuracy: {polarity_accuracy:.4f}")
    print(f"\nweighted Aspect category F1 score: {aspect_category_f1:.4f}")
    print(f"weighted Polarity F1 score: {polarity_f1:.4f}")
    
    print(f"\nMicro Aspect category F1 score: {micro_aspect_category_f1:.4f}")
    print(f"Micro Polarity F1 score: {micro_polarity_f1:.4f}")
    
    
    print(f"\nMacro Aspect category F1 score: {mac_aspect_category_f1:.4f}")
    print(f"Macro Polarity F1 score: {mac_polarity_f1:.4f}")
    
    return aspect_category_accuracy, polarity_accuracy, aspect_category_f1, polarity_f1


_,_,test_aspect_category_acc, test_polarity_acc = test1(model, test_dataloader)






# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('asafaya/bert-base-arabic')

# Define the test sentence
test_sentence = "عم حدث التطبيق ماعم يتحدث عم يطلع جاري التنزيل وبضل هيك وقت طويل"

# Tokenize the test sentence
encoded_sentence = tokenizer.encode_plus(
    test_sentence,
    max_length=128,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    return_tensors='pt'
)

# Make a prediction using the trained model
_,aspect_category_logits, polarity_logits = model.forward(encoded_sentence['input_ids'], encoded_sentence['attention_mask'])
predicted_aspect_category = torch.argmax(aspect_category_logits).item()
predicted_polarity = torch.argmax(polarity_logits).item()

# Print the predicted aspect category and polarity
print("Predicted Aspect Category: {}".format(predicted_aspect_category))
print("Predicted Polarity: {}".format(predicted_polarity))


# int to 1D shape
predicted_aspect_category_array = np.array([predicted_aspect_category])
predicted_polarity_array = np.array([predicted_polarity])

print(test_sentence)
print(category_encoder.inverse_transform(predicted_aspect_category_array))
print(polarity_encoder.inverse_transform(predicted_polarity_array))








#Plots and figers
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
# Plot the training loss
axs[0, 0].plot(epoch_list, train_loss_list, label='Training Loss')
axs[0, 0].set_xlabel('Epoch')
axs[0, 0].set_ylabel('Loss')
axs[0, 0].set_title('Training Loss over Epochs')
axs[0, 0].legend()

# Plot the validation loss
axs[0, 1].plot(epoch_list, val_loss_list, label='Validation Loss')
axs[0, 1].set_xlabel('Epoch')
axs[0, 1].set_ylabel('Loss')
axs[0, 1].set_title('Validation Loss over Epochs')
axs[0, 1].legend()

# Plot the category accuracy
axs[1, 0].plot(epoch_list, cat_acc_list, label='Category Accuracy')
axs[1, 0].set_xlabel('Epoch')
axs[1, 0].set_ylabel('Accuracy')
axs[1, 0].set_title('Category Accuracy over Epochs')
axs[1, 0].legend()

# Plot the polarity accuracy
axs[1, 1].plot(epoch_list, pol_acc_list, label='Polarity Accuracy')
axs[1, 1].set_xlabel('Epoch')
axs[1, 1].set_ylabel('Accuracy')
axs[1, 1].set_title('Polarity Accuracy over Epochs')
axs[1, 1].legend()

plt.show()
