# Lab3 code
#  Αλεξανδρόπουλος Σταμάτης 03117060, Γκότση Πολυτίμη-Άννα 03117201

import numpy as np
import os
import random
import gzip
import librosa.display
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.metrics import classification_report
import torch.optim as optim

"""**Βήμα 0**"""

os.listdir("../input/patreco3-multitask-affective-music/data/")

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

"""**Βήμα 1**

α)
"""

with open('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train_labels.txt', 'r') as f:
        lines = f.readlines()
        
        index1=random.randint(0,len(lines)-1)
        line1=lines[index1]
        spec_file1=line1.split()[0]
        label1=line1.split()[1]

        index2=random.randint(0,len(lines)-1)
        line2=lines[index2]
        spec_file2=line2.split()[0]
        label2=line2.split()[1]
        while label2==label1:
            index2=random.randint(0,len(lines)-1)
            line2=lines[index2]
            spec_file2=line2.split()[0]
            label2=line2.split()[1]
            
        line2=lines[index2]
        print('Labels chosen:', label1, label2)
        print('Spec files chosen:', spec_file1, spec_file2)

"""β)"""

spectrogram_file_1 = '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/'+spec_file1[:-3]
spectrogram_file_2 = '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/train/'+spec_file2[:-3]

spectrograms_1 = np.load(spectrogram_file_1)
spectrograms_2 = np.load(spectrogram_file_2)


# spectrograms contains a fused mel spectrogram and chromagram
# Decompose as follows
mel_spectrogram_1 = spectrograms_1[:128]
chromagram_1 = spectrograms_1[128:]
mel_spectrogram_2 = spectrograms_2[:128]
chromagram_2 = spectrograms_2[128:]

print(mel_spectrogram_1.shape)
print(chromagram_1.shape)
print(mel_spectrogram_2.shape)
print(chromagram_2.shape)

"""γ)"""

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(22,8))
img = librosa.display.specshow(mel_spectrogram_1, x_axis='time', y_axis='linear', ax=ax1)
ax1.set(title=spec_file1+": "+label1)
fig.colorbar(img, ax=ax1, format="%+2.f dB")
img = librosa.display.specshow(mel_spectrogram_2, x_axis='time', y_axis='linear', ax=ax2)
ax2.set(title=spec_file2+": "+label2)
fig.colorbar(img, ax=ax2, format="%+2.f dB")

"""**Βήμα 2**

α)
"""

print('Shape of spectrogram for '+label1+':',mel_spectrogram_1.shape)
print('Shape of spectrogram for '+label2+':',mel_spectrogram_2.shape)

"""β)"""

spectrogram_file_1 = '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/'+spec_file1[:-3]
spectrogram_file_2 = '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/train/'+spec_file2[:-3]

spectrograms_1_beat = np.load(spectrogram_file_1)
spectrograms_2_beat = np.load(spectrogram_file_2)


# spectrograms contains a fused mel spectrogram and chromagram
# Decompose as follows
mel_spectrogram_1_beat = spectrograms_1_beat[:128]
chromagram_1_beat = spectrograms_1_beat[128:]
mel_spectrogram_2_beat = spectrograms_2_beat[:128]
chromagram_2_beat = spectrograms_2_beat[128:]

print(mel_spectrogram_1_beat.shape)
print(chromagram_1_beat.shape)
print(mel_spectrogram_2_beat.shape)
print(chromagram_2_beat.shape)

print('Shape of beat-synced spectrogram for '+label1+':',mel_spectrogram_1_beat.shape)
print('Shape of beat-synced spectrogram for '+label2+':',mel_spectrogram_2_beat.shape)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(22,8))
img = librosa.display.specshow(mel_spectrogram_1_beat, x_axis='time', y_axis='linear', ax=ax1)
ax1.set(title=spec_file1+": "+label1)
fig.colorbar(img, ax=ax1, format="%+2.f dB")
img = librosa.display.specshow(mel_spectrogram_2_beat, x_axis='time', y_axis='linear', ax=ax2)
ax2.set(title=spec_file2+": "+label2)
fig.colorbar(img, ax=ax2, format="%+2.f dB")

"""**Βήμα 3**"""

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(22,8))
img = librosa.display.specshow(chromagram_1, x_axis='time', y_axis='linear', ax=ax1)
ax1.set(title=spec_file1+": "+label1)
fig.colorbar(img, ax=ax1, format="%+2.f dB")
img = librosa.display.specshow(chromagram_2, x_axis='time', y_axis='linear', ax=ax2)
ax2.set(title=spec_file2+": "+label2)
fig.colorbar(img, ax=ax2, format="%+2.f dB")

print('Shape of chromagram for '+label1+':',chromagram_1.shape)
print('Shape of chromagram for '+label2+':',chromagram_2.shape)

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(22,8))
img = librosa.display.specshow(chromagram_1_beat, x_axis='time', y_axis='linear', ax=ax1)
ax1.set(title=spec_file1+": "+label1)
fig.colorbar(img, ax=ax1, format="%+2.f dB")
img = librosa.display.specshow(chromagram_2_beat, x_axis='time', y_axis='linear', ax=ax2)
ax2.set(title=spec_file2+": "+label2)
fig.colorbar(img, ax=ax2, format="%+2.f dB")

print('Shape of beat-synced chromagram for '+label1+':',chromagram_1_beat.shape)
print('Shape of beat-synced chromagram for '+label2+':',chromagram_2_beat.shape)

"""**Βήμα 4**

γ)
"""

# this cell contains code provided by lab file: dataset.py, with two added functions

import copy
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler

# HINT: Use this class mapping to merge similar classes and ignore classes that do not work very well
CLASS_MAPPING = {
    "Rock": "Rock",
    "Psych-Rock": "Rock",
    "Indie-Rock": None,
    "Post-Rock": "Rock",
    "Psych-Folk": "Folk",
    "Folk": "Folk",
    "Metal": "Metal",
    "Punk": "Metal",
    "Post-Punk": None,
    "Trip-Hop": "Trip-Hop",
    "Pop": "Pop",
    "Electronic": "Electronic",
    "Hip-Hop": "Hip-Hop",
    "Classical": "Classical",
    "Blues": "Blues",
    "Chiptune": "Electronic",
    "Jazz": "Jazz",
    "Soundtrack": None,
    "International": None,
    "Old-Time": None,
}


def torch_train_val_split(
    dataset, batch_train, batch_eval, val_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    return train_loader, val_loader


def read_spectrogram(spectrogram_file, chroma=True):
    # with open(spectrogram_file, "r") as f:
    spectrograms = np.load(spectrogram_file)
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T

# __________________________________additions_______________________________________

def read_mel_spectrogram(spectrogram_file, chroma=True):
    # with open(spectrogram_file, "r") as f:
    spectrograms = np.load(spectrogram_file)[:128]
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T

def read_chromagram(spectrogram_file, chroma=True):
    # with open(spectrogram_file, "r") as f:
    spectrograms = np.load(spectrogram_file)[128:]
    # spectrograms contains a fused mel spectrogram and chromagram
    # Decompose as follows
    return spectrograms.T
#___________________________________________________________________________________

class LabelTransformer(LabelEncoder):
    def inverse(self, y):
        try:
            return super(LabelTransformer, self).inverse_transform(y)
        except:
            return super(LabelTransformer, self).inverse_transform([y])

    def transform(self, y):
        try:
            return super(LabelTransformer, self).transform(y)
        except:
            return super(LabelTransformer, self).transform([y])


class PaddingTransform(object):
    def __init__(self, max_length, padding_value=0):
        self.max_length = max_length
        self.padding_value = padding_value

    def __call__(self, s):
        if len(s) == self.max_length:
            return s

        if len(s) > self.max_length:
            return s[: self.max_length]

        if len(s) < self.max_length:
            s1 = copy.deepcopy(s)
            pad = np.zeros((self.max_length - s.shape[0], s.shape[1]), dtype=np.float32)
            s1 = np.vstack((s1, pad))
            return s1


class SpectrogramDataset(Dataset):
    def __init__(
        self, path, class_mapping=None, train=True, max_length=-1, regression=None, spect_type='spectrogram'
    ):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.regression = regression
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        self.files, labels = self.get_files_labels(self.index, class_mapping)

        if spect_type=='spectrogram':
            self.feats = [read_spectrogram(os.path.join(p, f)) for f in self.files]
        elif spect_type=='mel_spectrogram':
            self.feats = [read_mel_spectrogram(os.path.join(p, f)) for f in self.files]
        else:
            self.feats = [read_chromagram(os.path.join(p, f)) for f in self.files]
            
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)
        self.label_transformer = LabelTransformer()
        if isinstance(labels, (list, tuple)):
            if not regression:
                self.labels = np.array(
                    self.label_transformer.fit_transform(labels)
                ).astype("int64")
            else:
                self.labels = np.array(labels).astype("float64")

    def get_files_labels(self, txt, class_mapping):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split("\t") for l in fd.readlines()[1:]]
        files, labels = [], []
        for l in lines:
            if self.regression:
                l = l[0].split(",")
                files.append(l[0] + ".fused.full.npy")
                labels.append(l[self.regression])
                continue
            label = l[1]
            if class_mapping:
                label = class_mapping[l[1]]
            if not label:
                continue
            fname = l[0]
            if fname.endswith(".gz"):
                fname = ".".join(fname.split(".")[:-1])
            #if not train:
            #    temp = l[0].split(".")
            #    fname=temp[0] + ".fused.full.npy"
            
            temp = l[0].split(".")
            fname=temp[0] + ".fused.full.npy"
            
            files.append(fname)
            labels.append(label)
        return files, labels

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length

    def __len__(self):
        return len(self.labels)

# this cell contains code provided by lab file: dataset.py

dataset = SpectrogramDataset(
        '../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING, train=True
    )

print(dataset[10])
print(f"Input: {dataset[10][0].shape}")
print(f"Label: {dataset[10][1]}")
print(f"Original length: {dataset[10][2]}")

# create dataset instances with and without mapping:
dataset_before_mapping=SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', train=True)
dataset_after_mapping=SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING, train=True)

# find all labels in each mapping:

labels_before_mapping=[]
for item in dataset_before_mapping:
    labels_before_mapping.append(item[1])

labels_after_mapping=[]
for item in dataset_after_mapping:
    labels_after_mapping.append(item[1])

print(len(set(labels_after_mapping)))

# create histograms:

f, (ax1,ax2) = plt.subplots(1,2,figsize=(18,6))
plt.title('Histograms of labels', fontsize=22, pad=40)

ax1.set_title('Histogram before class mapping', fontsize=18)
ax1.hist(labels_before_mapping, bins=len(set(labels_before_mapping)))

ax2.set_title('Histogram after class mapping', fontsize=18)
ax2.hist(labels_after_mapping, bins=len(set(labels_after_mapping)))

"""**Βήμα 5**"""

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, rnn_size, output_dim, num_layers, bidirectional=False, dropout=0):
        super(BasicLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.feature_size = rnn_size * 2 if self.bidirectional else rnn_size
        self.num_layers = num_layers
        self.hidden_size=rnn_size
        self.dropout=dropout

        # --------------- Insert your code here ---------------- #
        # Initialize the LSTM, Dropout, Output layers

        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=rnn_size, num_layers=num_layers, batch_first=True, bidirectional=self.bidirectional, dropout=self.dropout) #lstm
        self.linear = nn.Linear(self.feature_size, output_dim)

    def forward(self, x, lengths):
        """ 
            x : 3D numpy array of dimension N x L x D
                N: batch index
                L: sequence index
                D: feature index

            lengths: N x 1
         """
        
        # --------------- Insert your code here ---------------- #
        
        # You must have all of the outputs of the LSTM, but you need only the last one (that does not exceed the sequence length)
        # To get it use the last_timestep method
        # Then pass it through the remaining network
        
        if not self.bidirectional:
            factor=1
        else:
            factor=2
            
#         h_0 = torch.zeros(self.num_layers*factor, x.size(0), self.hidden_size) #hidden state
#         c_0 = torch.zeros(self.num_layers*factor, x.size(0), self.hidden_size) #internal state
        
        x_packed = pack_padded_sequence(x.cpu(), lengths.cpu(), batch_first = True, enforce_sorted = False).to(device) 
        output, (hn, cn) = self.lstm(x_packed)  #lstm with input, hidden, and internal state
        output_unpacked,_ = pad_packed_sequence(output,batch_first = True)
        
        last_timestep_output=self.last_timestep(output_unpacked, lengths, self.bidirectional)
        last_outputs = self.linear(last_timestep_output)
        
        return last_outputs

    def last_timestep(self, outputs, lengths, bidirectional=False):
        """
            Returns the last output of the LSTM taking into account the zero padding
        """
        if bidirectional:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            # Concatenate and return - maybe add more functionalities like average
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

"""###3: Train LSTM, print only training loss per epoch"""

def train_model(model,optimizer,criterion,num_epochs,train_loader,validation_loader=None,validate=False,device="cuda", overfit_batch=False,patience=4,early_stopping=True):
  
  # _____train model:_____
    
    model.to(device)
    train_losses=[]                                          # array to hold mean train losses per epoch
    validation_losses=[]
    
    last_loss = 1000                                         # variable to hold loss of previous epoch, initialize to abig number
    trigger_times = 0                                        # number of times loss has not decreased
    epochs_performed=0
    
    if overfit_batch:                                        # if we want o overfit with training in one batch
        overfit_train_batch=next(iter(train_loader))         # get one batch from train loader
    
    for j in range(num_epochs):
        epochs_performed+=1                                  # increase number of epochs for which we have trained the model 
        train_loss_batch = []                                # array to hold train losses per batch

        if not overfit_batch:                                # normal training
            for i, data in enumerate(train_loader,0):        # for all batches
                if device:                                   # if we have GPU available
                    X=data[0].float().cuda()
                    y=data[1].cuda()
                    length=data[2].cuda()
                else:                                        # we are running on CPU
                    print('CPU')           
                    X=data[0].float()
                    y=data[1]
                    lenght=data[2]

                optimizer.zero_grad()                         # set gradients to zero

                outputs = model(X,length)                     # give input to model and get output                    

                loss = criterion(outputs, y)                  # calculate loss 

                loss.backward()                               # compute gradient   
                optimizer.step()                              # update parameters
                train_loss_batch.append(loss.item())          # add batch loss to array 
        else:                                                 # we are overfitting by training in only one batch
            data=overfit_train_batch                          # our data is the one batch
            if device:
                X=data[0].float().cuda()
                y=data[1].cuda()
                length=data[2].cuda()
            else:
                print('CPU')
                X=data[0].float()
                y=data[1]
                lenght=data[2]

            optimizer.zero_grad()                          # set gradients to zero

            outputs = model(X,length)                      # give input to model and get output                    

            loss = criterion(outputs, y)                   # calculate loss 

            loss.backward()                                # compute gradient   
            optimizer.step()                               # update parameters
            train_loss_batch.append(loss.item())           # add batch loss to array
            

        mean_training_loss=np.mean(train_loss_batch)           # calculate average training loss for epoch 
        train_losses.append(mean_training_loss)

        # _____evaluate on validation set:_____

        if validate and not overfit_batch:                     # in the case of overfitting with one batch we do not perform validation
            validation_loss_batch=[]                           # array to hold validation losses per batch
            with torch.no_grad():                              # disable gradient calculation as we will not call torch.backward()
                model.eval()

                for i, val_data in enumerate(validation_loader,0):
                    if device:
                        X_val=val_data[0].float().cuda()
                        y_val=val_data[1].cuda()
                        length_val=val_data[2].cuda()
                    else:
                        X_val=val_data[0].float()
                        y_val=val_data[1]
                        lenght_val=val_data[2]

                    pred = model(X_val,length_val)                 # get predictions
                    loss = criterion(pred,y_val)                   # calculate loss 
                    validation_loss_batch.append(loss.item())      # add batch loss to array

            model.train()                                          # train mode reset
            mean_validation_loss=np.mean(validation_loss_batch)    # calculate mean loss in epoch
            validation_losses.append(mean_validation_loss)
  
        print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
        if validate and not overfit_batch:
            print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
        print('--------------------------------------------------------------------------')
    
    
         # _____early stopping:_____ 
        
        if early_stopping and not overfit_batch:
            if mean_validation_loss >= last_loss:

                trigger_times += 1
                print('Number of times validation loss has not decreased:', trigger_times)
                print('--------------------------------------------------------------------------')

                if trigger_times >= patience:
                    print('Early stopping...')
                    if validate:
                        return model,train_losses, validation_losses, j+1
                    else:
                        return train_losses, j+1

            else:
                trigger_times = 0           
                torch.save(model.state_dict(), './model.pt')   # create checkpoint

            last_loss = mean_validation_loss
    
    
    if validate:
        return model,train_losses, validation_losses, j+1
    else:
        return train_losses, j+1

def evaluate(model, test_dataset, batch_size, device="cuda"):      # function to evaluate a trained model on a test set
   
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  # create dataloader from data
    model.eval()                                                   # we are performing evaluation
    
    y_pred=[]                                                      # array to hold our predictions
    y_real=[]                                                      # array to hold real labels
    
    for i, test_data in enumerate(test_loader,0):                  # for all batches
        if device:                                                 # we are using GPU
            X=test_data[0].float().cuda()
            y=test_data[1].cuda()
            length=test_data[2].cuda()
        else:                                                      # we are using CPU
            print('CPU')
            X=test_data[0].float()
            y=test_data[1]
            lenght=test_data[2]
        
        pred = model(X, length)                                    # get model predictions for batch
        y_predict=torch.argmax(pred,dim=1)                         # get maximum for each sample: the predicted class
        
        if device:                                                 # covert to lists, depending on GPU or CPU 
            y_predict= y_predict.data.cpu().numpy().tolist()
            y=y.data.cpu().numpy().tolist()
        else:
            y_predict=y_predict.data.numpy().tolist()
            y=y.data.numpy().tolist()
        
        y_pred+=y_predict                                          # add predictions to predictions from previuos batches
        y_real += y                                                # add labels to labels from previous batches
        
    print(classification_report(y_real, y_pred))                   # print evaluation metrics
    
    return

"""β)"""

#dataset split
batch_size = 32
val_size = 0.15
mels = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader_mel, validation_loader_mel = torch_train_val_split(mels, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 900
learning_rate = 1e-4
bidirectional = True 
dropout = 0.2
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader=train_loader_mel,validation_loader=validation_loader_mel,validate=True, early_stopping=early_stopping, patience=3, overfit_batch=True)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.title('Training loss per epoch for beat-synced mel spectrogram witn overfiting with one batch',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False,spect_type='mel_spectrogram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""γ)

**Απλά mel-spectrograms:**
"""

#dataset split
batch_size = 32
val_size = 0.15
mels = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader, validation_loader = torch_train_val_split(mels, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader=train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping, patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False,spect_type='mel_spectrogram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""δ)

**Beat-synced mel-spectrograms**
"""

#dataset split
batch_size = 32
val_size = 0.15
mels = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader, validation_loader = torch_train_val_split(mels, batch_size ,batch_size, val_size)

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader=train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping, patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False,spect_type='mel_spectrogram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""ε)

**Απλά chromagrams:**
"""

#dataset split
batch_size = 32
val_size = 0.15
chromas = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='chromagram')
train_loader, validation_loader= torch_train_val_split(chromas, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 12  # length of mel spectogram
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader=train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping, patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for chromagram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False,spect_type='chromagram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""**Beat-synced chromagrams:**"""

#dataset split
batch_size = 32
val_size = 0.15
chromas = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='chromagram')
train_loader, validation_loader = torch_train_val_split(chromas, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 12  # length of mel spectogram
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader=train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping, patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced chromagram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False,spect_type='chromagram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""ζ)

**Απλά, concatenated:**
"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 140  # 128+12 size
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for concatenated spectrograms and chromagrams',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""**Beat-synced concatenated**"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

#model parameters
rnn_dim = 64
input_dim = 140  # 128+12 size
output_dim = 10  # number of classes
num_layers = 2
num_epochs = 60
learning_rate = 1e-4
bidirectional = True 
dropout = 0.4
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
# model.to(device)
criterion = nn.CrossEntropyLoss()                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_model(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced concatenated spectrograms and chromagrams',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""**Βήμα 7**"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats

class CNN(nn.Module):
    def __init__(self,output_size, momentum,dropout):   # used for mel-spectrograms and fused spectrograms
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32,momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(128, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(256, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 4),
        )
        
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

class CNN(nn.Module):                                        # used only for chromagrams
    def __init__(self,output_size, momentum,dropout):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32,momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(64, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size = 3, padding = 3),
            nn.BatchNorm2d(128, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128,256, kernel_size = 3, padding = 3),
            nn.BatchNorm2d(256, momentum = momentum),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2),
        )
        
        self.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, output_size),
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        return self.final(x)

"""###3: Train CNN, print only training loss per epoch"""

def train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=None,validate=False,device="cuda", overfit_batch=False,patience=4,early_stopping=True):
  
  # _____train model:_____
    
    model.to(device)
    train_losses=[]                                          # array to hold mean train losses per epoch
    validation_losses=[]
    
    last_loss = 1000                                         # variable to hold loss of previous epoch, initialize to abig number
    trigger_times = 0                                        # number of times loss has not decreased
    epochs_performed=0
    
    if overfit_batch:                                        # if we want o overfit with training in one batch
        overfit_train_batch=next(iter(train_loader))         # get one batch from train loader
    
    for j in range(num_epochs):
        epochs_performed+=1                                  # increase number of epochs for which we have trained the model 
        train_loss_batch = []                                # array to hold train losses per batch

        if not overfit_batch:                                # normal training
            for i, data in enumerate(train_loader,0):        # for all batches
                if device:                                   # if we have GPU available
                    X=data[0].float().cuda()
                    y = data[1].cuda()
                    length=data[2].cuda()
                else:                                        # we are running on CPU
                    print('CPU')           
                    X=data[0].float()
                    y = data[1]
                    lenght=data[2]
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                optimizer.zero_grad()                         # set gradients to zero
                outputs = model(X)                     # give input to model and get output                    
                
                loss = criterion(outputs, y)                  # calculate loss 
                loss.backward()                               # compute gradient   
                optimizer.step()                              # update parameters
                train_loss_batch.append(loss.item())          # add batch loss to array 
        else:                                                 # we are overfitting by training in only one batch
            data=overfit_train_batch                          # our data is the one batch
            if device:
                X=data[0].float().cuda()
                y = data[1].cuda()
                length=data[2].cuda()
            else:
                print('CPU')
                X=data[0].float()
                y = data[1]
                lenght=data[2]
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            optimizer.zero_grad()                          # set gradients to zero

            outputs = model(X)                      # give input to model and get output                    

            loss = criterion(outputs, y)                   # calculate loss 

            loss.backward()                                # compute gradient   
            optimizer.step()                               # update parameters
            train_loss_batch.append(loss.item())           # add batch loss to array
            

        mean_training_loss=np.mean(train_loss_batch)           # calculate average training loss for epoch 
        train_losses.append(mean_training_loss)

        # _____evaluate on validation set:_____

        if validate and not overfit_batch:                     # in the case of overfitting with one batch we do not perform validation
            validation_loss_batch=[]                           # array to hold validation losses per batch
            with torch.no_grad():                              # disable gradient calculation as we will not call torch.backward()
                model.eval()

                for i, val_data in enumerate(validation_loader,0):
                    if device:
                        X_val=val_data[0].float().cuda()
                        y_val = val_data[1].cuda()
                        length_val= val_data[2].cuda()
                    else:
                        print('CPU')
                        X=val_data[0].float()
                        y_val = val_data[1]
                        length_val= val_data[2]
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
                    pred = model(X_val)                 # get predictions
                    loss = criterion(pred,y_val)                   # calculate loss 
                    validation_loss_batch.append(loss.item())      # add batch loss to array

            model.train()                                          # train mode reset
            mean_validation_loss=np.mean(validation_loss_batch)    # calculate mean loss in epoch
            validation_losses.append(mean_validation_loss)
  
        print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
        if validate and not overfit_batch:
            print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
        print('--------------------------------------------------------------------------')
    
    
         # _____early stopping:_____ 
        
        if early_stopping and not overfit_batch:
            if mean_validation_loss >= last_loss:

                trigger_times += 1
                print('Number of times validation loss has not decreased:', trigger_times)
                print('--------------------------------------------------------------------------')

                if trigger_times >= patience:
                    print('Early stopping...')
                    if validate:
                        return model,train_losses, validation_losses, j+1
                    else:
                        return train_losses, j+1

            else:
                trigger_times = 0           
                torch.save(model.state_dict(), './model.pt')   # create checkpoint

            last_loss = mean_validation_loss
    
    
    if validate:
        return model,train_losses, validation_losses, j+1
    else:
        return train_losses, j+1

def evaluate(model, test_dataset, batch_size, device="cuda"):      # function to evaluate a trained model on a test set
   
    test_loader = DataLoader(test_dataset, batch_size=batch_size)  # create dataloader from data
    model.eval()                                                   # we are performing evaluation
    
    y_pred=[]                                                      # array to hold our predictions
    y_real=[]                                                      # array to hold real labels
    
    for i, test_data in enumerate(test_loader,0):                  # for all batches
        if device:                                                 # we are using GPU
            X=test_data[0].float().cuda()
            y=test_data[1].cuda()
            length=test_data[2].cuda()
        else:                                                      # we are using CPU
            print('CPU')
            X=test_data[0].float()
            y=test_data[1]
            lenght=test_data[2]
            
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        pred = model(X)                                    # get model predictions for batch
        y_predict=torch.argmax(pred,dim=1)                         # get maximum for each sample: the predicted class
        
        if device:                                                 # covert to lists, depending on GPU or CPU 
            y_predict= y_predict.data.cpu().numpy().tolist()
            y=y.data.cpu().numpy().tolist()
        else:
            y_predict=y_predict.data.numpy().tolist()
            y=y.data.numpy().tolist()
        
        y_pred+=y_predict                                          # add predictions to predictions from previuos batches
        y_real += y                                                # add labels to labels from previous batches
        
    print(classification_report(y_real, y_pred))                   # print evaluation metrics
    
    return

"""**δ)**"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 300
learning_rate = 1e-4
dropout = 0.05
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3,overfit_batch=True)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.title('Training loss per epoch for spectrograms witn overfiting with one batch',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""**ε)**

Mel-spectrograms
"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for mel spectrograms',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False,spect_type='mel_spectrogram',max_length=spects.max_length)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""Mel spectrograms, Beat-synced"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced mel spectrograms',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False, max_length=spects.max_length, spect_type='mel_spectrogram')
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""Chromagrams"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='chromagram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for chromagrams',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False,spect_type='chromagram', max_length=spects.max_length)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""Beat synced chromagrams"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='chromagram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced chromagrams',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False,spect_type='chromagram', max_length=spects.max_length)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""Απλά, concatenated"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for concatenated spectrograms',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/', class_mapping=CLASS_MAPPING,train=False, spect_type='spectrogram', max_length=spects.max_length)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""Beat-synced concatenated"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/',class_mapping=CLASS_MAPPING,train=True,spect_type='spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for beat-synced concatenated spectrograms',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
test_dataset = SpectrogramDataset('/kaggle/input/patreco3-multitask-affective-music/data/fma_genre_spectrograms_beat/', class_mapping=CLASS_MAPPING,train=False, spect_type='spectrogram', max_length=spects.max_length)
batch_train = 32
evaluate(model, test_dataset ,batch_train)

"""**Βήμα 8**"""

def torch_train_val_test_split(
    dataset, batch_train, batch_eval, batch_test, val_size=0.2, test_size=0.2, shuffle=True, seed=420
):
    # Creating data indices for training-validation and test splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    test_split = int(np.floor(test_size * dataset_size))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(indices)
    temp_indices = indices[test_split:]
    test_indices = indices[:test_split]
     
    # Creating data indices for training and validation splits:
    val_split = int(np.floor(val_size * (dataset_size-len(test_indices))))
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(temp_indices)
    train_indices=temp_indices[val_split:]
    val_indices=temp_indices[:val_split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler= SubsetRandomSampler(test_indices)

    train_loader = DataLoader(dataset, batch_size=batch_train, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=batch_eval, sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_size=batch_test, sampler=test_sampler)
    return train_loader, val_loader, test_loader

class SpectrogramDatasetMulti(Dataset):
    def __init__(
        self, path, train=True, max_length=-1, label_type=None
    ):
        t = "train" if train else "test"
        p = os.path.join(path, t)
        self.label_type=label_type
        self.index = os.path.join(path, "{}_labels.txt".format(t))
        if self.label_type!=None:
            self.files, labels = self.get_files_labels(self.index)
            if isinstance(labels, (list, tuple)):
                self.labels = np.array(labels).astype("float64")
        else:    
            self.files, self.valence, self.energy, self.danceability = self.get_files_labels(self.index)
            if isinstance(self.valence, (list, tuple)):
                self.valence = np.array(self.valence).astype("float64")
            if isinstance(self.energy, (list, tuple)):
                self.energy = np.array(self.energy).astype("float64")
            if isinstance(self.danceability, (list, tuple)):
                self.danceability = np.array(self.danceability).astype("float64")
        self.feats = [read_mel_spectrogram(os.path.join(p, f)) for f in self.files]
        self.feat_dim = self.feats[0].shape[1]
        self.lengths = [len(i) for i in self.feats]
        self.max_length = max(self.lengths) if max_length <= 0 else max_length
        self.zero_pad_and_stack = PaddingTransform(self.max_length)

    def get_files_labels(self, txt):
        with open(txt, "r") as fd:
            lines = [l.rstrip().split(",") for l in fd.readlines()[1:]]
        if self.label_type!=None:
            files, labels = [], []
        else:
            files, labels = [], [[],[],[]]
        for l in lines:
            if self.label_type=='valence':
                label=l[1]
            elif self.label_type== 'energy':
                label=l[2]
            elif self.label_type=='danceability':
                label=l[3]
            else:
                label=[l[1],l[2],l[3]]
            if not label:
                continue
            fname = l[0]
            fname= fname + ".fused.full.npy"
            files.append(fname)
            if self.label_type!=None:
                labels.append(label)
            else:
                labels[0].append(l[1])
                labels[1].append(l[2])
                labels[2].append(l[3])
        if self.label_type!=None:
            return files, labels
        else:
            return files, labels[0],labels[1],labels[2]

    def __getitem__(self, item):
        length = min(self.lengths[item], self.max_length)
        if self.label_type!=None:
            return self.zero_pad_and_stack(self.feats[item]), self.labels[item], length
        else:
            return self.zero_pad_and_stack(self.feats[item]), self.valence[item], self.energy[item], self.danceability[item] ,length

    def __len__(self):
#         return len(self.labels)
        return len(self.files)

"""**α)**

LSTM:
"""

"""###3: Train LSTM, print only training loss per epoch"""

def train_LSTM_model_reg(model,optimizer,criterion,num_epochs,train_loader,validation_loader=None,validate=False,device="cuda", overfit_batch=False,patience=4,early_stopping=True):
  
  # _____train model:_____
    
    model.to(device)
    train_losses=[]                                          # array to hold mean train losses per epoch
    validation_losses=[]
    
    last_loss = 1000                                         # variable to hold loss of previous epoch, initialize to abig number
    trigger_times = 0                                        # number of times loss has not decreased
    epochs_performed=0
    
    if overfit_batch:                                        # if we want o overfit with training in one batch
        overfit_train_batch=next(iter(train_loader))         # get one batch from train loader
    
    for j in range(num_epochs):
        epochs_performed+=1                                  # increase number of epochs for which we have trained the model 
        train_loss_batch = []                                # array to hold train losses per batch

        if not overfit_batch:                                # normal training
            for i, data in enumerate(train_loader,0):        # for all batches
                if device:                                   # if we have GPU available
                    X=data[0].float().cuda()
                    y=data[1].cuda()
                    length=data[2].cuda()
                else:                                        # we are running on CPU
                    print('CPU')           
                    X=data[0].float()
                    y=data[1]
                    lenght=data[2]

                optimizer.zero_grad()                         # set gradients to zero

                outputs = model(X,length)                     # give input to model and get output   
                outputs=torch.reshape(outputs,(-1,))

                loss = criterion(outputs, y.float())                  # calculate loss 

                loss.backward()                               # compute gradient   
                optimizer.step()                              # update parameters
                train_loss_batch.append(loss.item())          # add batch loss to array 
        else:                                                 # we are overfitting by training in only one batch
            data=overfit_train_batch                          # our data is the one batch
            if device:
                X=data[0].float().cuda()
                y=data[1].cuda()
                length=data[2].cuda()
            else:
                print('CPU')
                X=data[0].float()
                y=data[1]
                lenght=data[2]

            optimizer.zero_grad()                          # set gradients to zero

            outputs = model(X,length)                      # give input to model and get output 
            outputs=torch.reshape(outputs,(-1,))

            loss = criterion(outputs, y.float())                   # calculate loss 

            loss.backward()                                # compute gradient   
            optimizer.step()                               # update parameters
            train_loss_batch.append(loss.item())           # add batch loss to array
            

        mean_training_loss=np.mean(train_loss_batch)           # calculate average training loss for epoch 
        train_losses.append(mean_training_loss)

        # _____evaluate on validation set:_____

        if validate and not overfit_batch:                     # in the case of overfitting with one batch we do not perform validation
            validation_loss_batch=[]                           # array to hold validation losses per batch
            with torch.no_grad():                              # disable gradient calculation as we will not call torch.backward()
                model.eval()

                for i, val_data in enumerate(validation_loader,0):
                    if device:
                        X_val=val_data[0].float().cuda()
                        y_val=val_data[1].cuda()
                        length_val=val_data[2].cuda()
                    else:
                        X_val=val_data[0].float()
                        y_val=val_data[1]
                        lenght_val=val_data[2]

                    pred = model(X_val,length_val)                 # get predictions
                    pred=torch.reshape(pred,(-1,))
                    loss = criterion(pred,y_val.float())                   # calculate loss 
                    validation_loss_batch.append(loss.item())      # add batch loss to array

            model.train()                                          # train mode reset
            mean_validation_loss=np.mean(validation_loss_batch)    # calculate mean loss in epoch
            validation_losses.append(mean_validation_loss)
  
        print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
        if validate and not overfit_batch:
            print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
        print('--------------------------------------------------------------------------')
    
    
         # _____early stopping:_____ 
        
        if early_stopping and not overfit_batch:
            if mean_validation_loss >= last_loss:

                trigger_times += 1
                print('Number of times validation loss has not decreased:', trigger_times)
                print('--------------------------------------------------------------------------')

                if trigger_times >= patience:
                    print('Early stopping...')
                    if validate:
                        return model,train_losses, validation_losses, j+1
                    else:
                        return train_losses, j+1

            else:
                trigger_times = 0           
                torch.save(model.state_dict(), './model_reg.pt')   # create checkpoint

            last_loss = mean_validation_loss
    
    
    if validate:
        return model,train_losses, validation_losses, j+1
    else:
        return train_losses, j+1

def evaluate_LSTM_reg(model, test_loader, batch_size, device="cuda"):      # function to evaluate a trained model on a test set
   

    model.eval()                                                   # we are performing evaluation
    
    y_pred=[]                                                      # array to hold our predictions
    y_real=[]                                                      # array to hold real labels
    
    for i, test_data in enumerate(test_loader,0):                  # for all batches
        if device:                                                 # we are using GPU
            X=test_data[0].float().cuda()
            y=test_data[1].cuda()
            length=test_data[2].cuda()
        else:                                                      # we are using CPU
            print('CPU')
            X=test_data[0].float()
            y=test_data[1]
            lenght=test_data[2]
        
        y_predict = model(X, length)                                    # get model predictions for batch
        
        if device:                                                 # covert to lists, depending on GPU or CPU 
            y_predict= y_predict.data.cpu().numpy().tolist()
            y=y.data.cpu().numpy().tolist()
        else:
            y_predict=y_predict.data.numpy().tolist()
            y=y.data.numpy().tolist()
        
        y_pred+=y_predict                                          # add predictions to predictions from previuos batches
        y_real += y                                                # add labels to labels from previous batches
    
    print("Spearman correlation:" ,stats.spearmanr(y_real, y_pred)[0])                   # print evaluation metrics
    
    return

"""CNN:"""

"""###3: Train CNN, print only training loss per epoch"""

def train_CNNmodel_reg(model,optimizer,criterion,num_epochs,train_loader,validation_loader=None,validate=False,device="cuda", overfit_batch=False,patience=4,early_stopping=True):
  
  # _____train model:_____
    
    model.to(device)
    train_losses=[]                                          # array to hold mean train losses per epoch
    validation_losses=[]
    
    last_loss = 1000                                         # variable to hold loss of previous epoch, initialize to abig number
    trigger_times = 0                                        # number of times loss has not decreased
    epochs_performed=0
    
    if overfit_batch:                                        # if we want o overfit with training in one batch
        overfit_train_batch=next(iter(train_loader))         # get one batch from train loader
    
    for j in range(num_epochs):
        epochs_performed+=1                                  # increase number of epochs for which we have trained the model 
        train_loss_batch = []                                # array to hold train losses per batch

        if not overfit_batch:                                # normal training
            for i, data in enumerate(train_loader,0):        # for all batches
                if device:                                   # if we have GPU available
                    X=data[0].float().cuda()
                    y = data[1].cuda()
                    length=data[2].cuda()
                else:                                        # we are running on CPU
                    print('CPU')           
                    X=data[0].float()
                    y = data[1]
                    lenght=data[2]
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
                optimizer.zero_grad()                         # set gradients to zero
                outputs = model(X)                     # give input to model and get output                    
                
                outputs=torch.reshape(outputs,(-1,))
                loss = criterion(outputs, y.float())                  # calculate loss 
                loss.backward()                               # compute gradient   
                optimizer.step()                              # update parameters
                train_loss_batch.append(loss.item())          # add batch loss to array 
        else:                                                 # we are overfitting by training in only one batch
            data=overfit_train_batch                          # our data is the one batch
            if device:
                X=data[0].float().cuda()
                y = data[1].cuda()
                length=data[2].cuda()
            else:
                print('CPU')
                X=data[0].float()
                y = data[1]
                lenght=data[2]
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
            optimizer.zero_grad()                          # set gradients to zero

            outputs = model(X)                      # give input to model and get output    
            outputs=torch.reshape(outputs,(-1,))

            loss = criterion(outputs, y.float())                   # calculate loss 

            loss.backward()                                # compute gradient   
            optimizer.step()                               # update parameters
            train_loss_batch.append(loss.item())           # add batch loss to array
            

        mean_training_loss=np.mean(train_loss_batch)           # calculate average training loss for epoch 
        train_losses.append(mean_training_loss)

        # _____evaluate on validation set:_____

        if validate and not overfit_batch:                     # in the case of overfitting with one batch we do not perform validation
            validation_loss_batch=[]                           # array to hold validation losses per batch
            with torch.no_grad():                              # disable gradient calculation as we will not call torch.backward()
                model.eval()

                for i, val_data in enumerate(validation_loader,0):
                    if device:
                        X_val=val_data[0].float().cuda()
                        y_val = val_data[1].cuda()
                        length_val= val_data[2].cuda()
                    else:
                        print('CPU')
                        X=val_data[0].float()
                        y_val = val_data[1]
                        length_val= val_data[2]
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
                    pred = model(X_val)                 # get predictions
                    pred=torch.reshape(pred,(-1,))
                    loss = criterion(pred,y_val.float())                   # calculate loss 
                    validation_loss_batch.append(loss.item())      # add batch loss to array

            model.train()                                          # train mode reset
            mean_validation_loss=np.mean(validation_loss_batch)    # calculate mean loss in epoch
            validation_losses.append(mean_validation_loss)
  
        print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
        if validate and not overfit_batch:
            print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
        print('--------------------------------------------------------------------------')
    
    
         # _____early stopping:_____ 
        
        if early_stopping and not overfit_batch:
            if mean_validation_loss >= last_loss:

                trigger_times += 1
                print('Number of times validation loss has not decreased:', trigger_times)
                print('--------------------------------------------------------------------------')

                if trigger_times >= patience:
                    print('Early stopping...')
                    if validate:
                        return model,train_losses, validation_losses, j+1
                    else:
                        return train_losses, j+1

            else:
                trigger_times = 0           
                torch.save(model.state_dict(), './model_reg.pt')   # create checkpoint

            last_loss = mean_validation_loss
    
    
    if validate:
        return model,train_losses, validation_losses, j+1
    else:
        return train_losses, j+1

def evaluate_CNN_reg(model, test_loader, batch_size, device="cuda"):      # function to evaluate a trained model on a test set
   
    model.eval()                                                   # we are performing evaluation
    
    y_pred=[]                                                      # array to hold our predictions
    y_real=[]                                                      # array to hold real labels
    
    for i, test_data in enumerate(test_loader,0):                  # for all batches
        if device:                                                 # we are using GPU
            X=test_data[0].float().cuda()
            y=test_data[1].cuda()
            length=test_data[2].cuda()
        else:                                                      # we are using CPU
            print('CPU')
            X=test_data[0].float()
            y=test_data[1]
            lenght=test_data[2]
            
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        y_predict = model(X)                                    # get model predictions for batch
        
        if device:                                                 # covert to lists, depending on GPU or CPU 
            y_predict= y_predict.data.cpu().numpy().tolist()
            y=y.data.cpu().numpy().tolist()
        else:
            y_predict=y_predict.data.numpy().tolist()
            y=y.data.numpy().tolist()
        
        y_pred+=y_predict                                          # add predictions to predictions from previuos batches
        y_real += y                                                # add labels to labels from previous batches
    
    print("Spearman correlation:" ,stats.spearmanr(y_real, y_pred)[0])                   # print evaluation metrics

    return

"""**β) Valence**"""

#dataset split
batch_size = 32
val_size = 0.10
test_size=0.10

specs_multi_valence = SpectrogramDatasetMulti(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         train=True, label_type='valence')

train_loader_val,validation_loader_val, test_loader_val = torch_train_val_test_split(specs_multi_valence, batch_size ,batch_size, batch_size, val_size, test_size)

"""LSTM"""

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 1   # number of classes
num_layers = 2
num_epochs = 20
learning_rate = 1e-4
bidirectional = True 
dropout = 0.2
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_LSTM_model_reg(model,optimizer,criterion,num_epochs,train_loader=train_loader_val,validation_loader=validation_loader_val,validate=True, early_stopping=early_stopping, patience=3, overfit_batch=False)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('LSTM: Training and validation loss per epoch for mel spectrogram, valence',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
batch_train = 32
evaluate_LSTM_reg(model, test_loader_val ,batch_train)

"""CNN"""

output_size = 1
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel_reg(model,optimizer,criterion,num_epochs,train_loader_val,validation_loader=validation_loader_val,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('CNN: Training and validation loss per epoch for mel spectrogram, valence',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
batch_train = 32
evaluate_CNN_reg(model, test_loader_val ,batch_train)

"""**γ) Energy**"""

#dataset split
batch_size = 32
val_size = 0.10
test_size=0.10

specs_multi_valence = SpectrogramDatasetMulti(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         train=True, label_type='energy')

train_loader_en,validation_loader_en, test_loader_en = torch_train_val_test_split(specs_multi_valence, batch_size ,batch_size, batch_size, val_size, test_size)

"""LSTM"""

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 1   # number of classes
num_layers = 2
num_epochs = 20
learning_rate = 1e-4
bidirectional = True 
dropout = 0.2
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_LSTM_model_reg(model,optimizer,criterion,num_epochs,train_loader=train_loader_en,validation_loader=validation_loader_en,validate=True, early_stopping=early_stopping, patience=3, overfit_batch=False)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('LSTM: Training and validation loss per epoch for mel spectrogram, energy',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
batch_train = 32
evaluate_LSTM_reg(model, test_loader_en ,batch_train)

"""CNN"""

output_size = 1
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel_reg(model,optimizer,criterion,num_epochs,train_loader_en,validation_loader=validation_loader_en,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('CNN: Training and validation loss per epoch for mel spectrogram, energy',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
evaluate_CNN_reg(model, test_loader_en ,batch_train)

"""**δ) Danceabilty**"""

#dataset split
batch_size = 32
val_size = 0.10
test_size=0.10

specs_multi_valence = SpectrogramDatasetMulti(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         train=True, label_type='danceability')

train_loader_dan,validation_loader_dan, test_loader_dan = torch_train_val_test_split(specs_multi_valence, batch_size ,batch_size, batch_size, val_size, test_size)

"""LSTM"""

#model parameters
rnn_dim = 64
input_dim = 128  # length of mel spectogram
output_dim = 1   # number of classes
num_layers = 2
num_epochs = 20
learning_rate = 1e-4
bidirectional = True 
dropout = 0.2
weight_decay = 1e-4
early_stopping = True

model = BasicLSTM(input_dim, rnn_dim, output_dim, num_layers, dropout=dropout)                      # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_LSTM_model_reg(model,optimizer,criterion,num_epochs,train_loader=train_loader_dan,validation_loader=validation_loader_dan,validate=True, early_stopping=early_stopping, patience=3, overfit_batch=False)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('RNN: Training and validation loss per epoch for mel spectrogram, danceability',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
batch_train = 32
evaluate_LSTM_reg(model, test_loader_dan ,batch_train)

"""CNN"""

output_size = 1
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.MSELoss().to(device)                                                                   # loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel_reg(model,optimizer,criterion,num_epochs,train_loader_dan,validation_loader=validation_loader_dan,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('CNN: Training and validation loss per epoch for mel spectrogram, danceability',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
evaluate_CNN_reg(model, test_loader_dan ,batch_train)

"""**Βήμα 9**

γ)

Load data for valence:
"""

#dataset split
batch_size = 32
val_size = 0.15
spects = SpectrogramDataset('../input/patreco3-multitask-affective-music/data/fma_genre_spectrograms/',class_mapping=CLASS_MAPPING,train=True,spect_type='mel_spectrogram')
train_loader, validation_loader = torch_train_val_split(spects, batch_size ,batch_size, val_size)

"""Define and train CNN model:"""

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.2
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)            # initialize model
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model, train_losses, validation_losses, epochs_run=train_CNNmodel(model,optimizer,criterion,num_epochs,train_loader,validation_loader=validation_loader,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('Training and validation loss per epoch for spectrograms',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

"""Load saved trained model and perform transfer learning:"""

model_trans = CNN(output_size,momentum,dropout)        # initialize model class
model_trans.load_state_dict(torch.load('./model.pt'))  # load trained model

for param in model_trans.parameters():              #freeze the parameters
    param.requires_grad = False

# reinitialize only last few layers (the linear layers) to fine-tune on task
model_trans.final = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LazyLinear(1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
    nn.ReLU(),
            nn.Linear(10, 1),
        )
print(model_trans)                     # print layers

#load data

#dataset split
batch_size = 32
val_size = 0.10
test_size=0.10

specs_multi_valence = SpectrogramDatasetMulti(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         train=True, label_type='valence')

train_loader_val,validation_loader_val, test_loader_val = torch_train_val_test_split(specs_multi_valence, batch_size ,batch_size, batch_size, val_size, test_size)

criterion = nn.MSELoss()                                                                   # loss function
optimizer = torch.optim.Adam(model_trans.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

# train (fine-tune) for a few epochs
num_epochs=10
model_trans, train_losses, validation_losses, epochs_run=train_CNNmodel_reg(model_trans,optimizer,criterion,num_epochs,train_loader_val,validation_loader=validation_loader_val,validate=True, early_stopping=early_stopping,patience=3)

# test performance
batch_train=32
evaluate_CNN_reg(model_trans, test_loader_val ,batch_train)

"""**Βήμα 10**"""

class Multitask_loss(nn.Module):
    def __init__(self, criterion, weights):
        super(Multitask_loss, self).__init__()
        self.criterion=criterion
        self.weights=weights
 
    def calculate_loss(self, logits, targets):        
        
        losses=[]                                           # list to hold losses
    
        for i in range (0,len(logits)):
            losses.append(self.criterion(logits[i],targets[i]))   # calculate loss

        final_loss = self.weights[0]* losses[0] + self.weights[1] * losses[1]+ self.weights[2] * losses[2]  # final loss is weighted sum of losses
        return final_loss,losses

"""###3: Train CNN, print only training loss per epoch"""

def train_CNNmodel_multi(model,optimizer,multitask_loss,num_epochs,train_loader,validation_loader=None,validate=False,device="cuda", overfit_batch=False,patience=4,early_stopping=True):
  
  # _____train model:_____
    
    model.to(device)
    train_losses, train_losses_valence,train_losses_energy,train_losses_danceability = [],[],[],[]                                          # array to hold mean train losses per epoch
    validation_losses, validation_losses_valence,validation_losses_energy,validation_losses_danceability = [],[],[],[]
    
    last_loss = 1000                                         # variable to hold loss of previous epoch, initialize to abig number
    trigger_times = 0                                        # number of times loss has not decreased
    epochs_performed=0
    
    if overfit_batch:                                        # if we want o overfit with training in one batch
        overfit_train_batch=next(iter(train_loader))         # get one batch from train loader
    
    for j in range(num_epochs):
        epochs_performed+=1                                  # increase number of epochs for which we have trained the model 
        train_loss_batch = [] 
        train_loss_batch_valence = []                                # array to hold train losses per batch
        train_loss_batch_energy = []
        train_loss_batch_danceability = [] 
        if not overfit_batch:                                # normal training
            for i, data in enumerate(train_loader,0):        # for all batches
                if device:                                   # if we have GPU available
                    X=data[0].float().cuda()
                    y_valence = data[1].cuda()
                    y_energy=data[2].cuda()
                    y_danceability= data[3].cuda()
                else:                                        # we are running on CPU
                    print('CPU')           
                    X=data[0].float()
                    y_valence = data[1]
                    y_energy=data[2]
                    y_danceability= data[3]
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).float()
                optimizer.zero_grad()                         # set gradients to zero
                outputs = model(X).float()                     # give input to model and get output                    
                
                loss,losses=multitask_loss.calculate_loss([outputs[:,0],outputs[:,1],outputs[:,2]],[y_valence.float(),y_energy.float(),y_danceability.float()])   # calculate loss
                
                loss_valence=losses[0]
                loss_energy=losses[1]
                loss_danceability=losses[2]   # get losses for printing
                 
                loss.backward()                               # compute gradient   
                optimizer.step()                              # update parameters
                train_loss_batch_valence.append(loss_valence.item()) # add batch loss_valence to array 
                train_loss_batch_energy.append(loss_energy.item()) # add batch loss_energy to array 
                train_loss_batch_danceability.append(loss_danceability.item()) # add batch loss_dancebility to array 
                train_loss_batch.append(loss.item())          # add batch loss to array 
                
                
        else:                                                 # we are overfitting by training in only one batch
            data=overfit_train_batch                          # our data is the one batch
            if device:                                   # if we have GPU available
                X=data[0].float().cuda()
                y_valence = data[1].cuda()
                y_energy=data[2].cuda()
                y_danceability= data[3].cuda()
            else:                                        # we are running on CPU
                print('CPU')           
                X=data[0].float()
                y_valence = data[1]
                y_energy=data[2]
                y_danceability= data[3]
            X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]).float()
            optimizer.zero_grad()                         # set gradients to zero
            outputs = model(X).float()                     # give input to model and get output                    
            
            loss,losses=multitask_loss.calculate_loss([outputs[:,0],outputs[:,1],outputs[:,2]],[y_valence.float(),y_energy.float(),y_danceability.float()])   # calculate loss
                
            loss_valence=losses[0]   # get losses for printing
            loss_energy=losses[1]
            loss_danceability=losses[2]  

            loss.backward()                               # compute gradient   
            optimizer.step()                              # update parameters
            train_loss_batch_valence.append(loss_valence.item()) # add batch loss_valence to array 
            train_loss_batch_energy.append(loss_energy.item()) # add batch loss_energy to array 
            train_loss_batch_danceability.append(loss_danceability.item()) # add batch loss_dancebility to array 
            train_loss_batch.append(loss.item())          # add batch loss to array 

        mean_training_loss=np.mean(train_loss_batch)           # calculate average training loss for epoch 
        mean_training_loss_valence =  np.mean(train_loss_batch_valence) 
        mean_training_loss_energy = np.mean(train_loss_batch_energy) 
        mean_training_loss_danceability = np.mean(train_loss_batch_danceability) 
        train_losses.append(mean_training_loss)
        train_losses_valence.append(mean_training_loss_valence)
        train_losses_energy.append(mean_training_loss_energy)
        train_losses_danceability.append(mean_training_loss_danceability)

        # _____evaluate on validation set:_____

        if validate and not overfit_batch:                     # in the case of overfitting with one batch we do not perform validation
            validation_loss_batch=[]                           # array to hold validation losses per batch
            validation_loss_batch_valence=[]
            validation_loss_batch_energy=[]
            validation_loss_batch_danceability=[]
            with torch.no_grad():                              # disable gradient calculation as we will not call torch.backward()
                model.eval()

                for i, val_data in enumerate(validation_loader,0):
                    if device:
                        X_val=val_data[0].float().cuda()
                        y_val_valence = val_data[1].cuda()
                        y_val_energy= val_data[2].cuda()
                        y_val_danceability= val_data[3].cuda()
                    else:
                        print('CPU')
                        X=val_data[0].float()
                        y_val_valence = val_data[1]
                        y_val_energy= val_data[2]
                        y_val_danceability= val_data[3]
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
                    pred = model(X_val)                 # get predictions
                    
                    loss,losses=multitask_loss.calculate_loss([outputs[:,0],outputs[:,1],outputs[:,2]],[y_valence.float(),y_energy.float(),y_danceability.float()])   # calculate loss
                
                    loss_valence=losses[0]  # get losses for printing
                    loss_energy=losses[1]
                    loss_danceability=losses[2]

                                        
                    validation_loss_batch_valence.append(loss_valence.item())      # add batch loss_valence to array
                    validation_loss_batch_energy.append(loss_energy.item())      # add batch loss_energy to array
                    validation_loss_batch_danceability.append(loss_danceability.item())      # add batch loss_danceability to array
                    validation_loss_batch.append(loss.item())      # add batch loss to array

            model.train()                                          # train mode reset
            mean_validation_loss=np.mean(validation_loss_batch)    # calculate mean loss in epoch
            mean_validation_loss_valence=np.mean(validation_loss_batch_valence)
            mean_validation_loss_energy=np.mean(validation_loss_batch_energy)
            mean_validation_loss_danceability=np.mean(validation_loss_batch_danceability)
            validation_losses.append(mean_validation_loss)
            validation_losses_valence.append(mean_validation_loss_valence)
            validation_losses_energy.append(mean_validation_loss_energy)
            validation_losses_danceability.append(mean_validation_loss_danceability)
            
  
        print("Epoch {}: Mean training loss per epoch: {}".format(j,mean_training_loss))
        print("Epoch {}: Mean training loss_valence per epoch: {}".format(j,mean_training_loss_valence))
        print("Epoch {}: Mean training loss_energy per epoch: {}".format(j,mean_training_loss_energy))
        print("Epoch {}: Mean training loss_danceability per epoch: {}".format(j,mean_training_loss_danceability))
        if validate and not overfit_batch:
            print("Epoch {}: Mean validation loss per epoch: {}".format(j,mean_validation_loss))
            print("Epoch {}: Mean validation loss_valence per epoch: {}".format(j,mean_validation_loss_valence))
            print("Epoch {}: Mean validation loss_energy per epoch: {}".format(j,mean_validation_loss_energy))
            print("Epoch {}: Mean validation loss_danceability per epoch: {}".format(j,mean_validation_loss_danceability))
        print('--------------------------------------------------------------------------')
    
    
         # _____early stopping:_____ 
        
        if early_stopping and not overfit_batch:
            if mean_validation_loss >= last_loss:

                trigger_times += 1
                print('Number of times validation loss has not decreased:', trigger_times)
                print('--------------------------------------------------------------------------')

                if trigger_times >= patience:
                    print('Early stopping...')
                    if validate:
                        return model,train_losses,train_losses_valence, train_losses_energy, train_losses_danceability,validation_losses, validation_losses_valence, validation_losses_energy, validation_losses_danceability, j+1
                    else:
                        return train_losses, j+1

            else:
                trigger_times = 0           
                torch.save(model.state_dict(), './model_reg.pt')   # create checkpoint

            last_loss = mean_validation_loss
    
    
    if validate:
        return model,train_losses,train_losses_valence, train_losses_energy, train_losses_danceability,validation_losses, validation_losses_valence, validation_losses_energy, validation_losses_danceability, j+1
    else:
        return train_losses, train_losses_valence, train_losses_energy, train_losses_danceability, j+1

def evaluate_CNN_multi(model, test_loader, batch_size, device="cuda"):      # function to evaluate a trained model on a test set
   
    model.eval()                                                   # we are performing evaluation
    
    y_pred,y_pred_valence,y_pred_energy,y_pred_danceability=[],[],[],[]                                                      # array to hold our predictions
    y_real,y_real_valence,y_real_energy,y_real_danceability=[],[],[],[]                                                      # array to hold real labels
    
    for i, test_data in enumerate(test_loader,0):                  # for all batches
        if device:                                                 # we are using GPU
            X=test_data[0].float().cuda()
            y_valence = test_data[1].cuda()
            y_energy = test_data[2].cuda()
            y_danceability = test_data[3].cuda()
        else:                                                      # we are using CPU
            print('CPU')
            X=test_data[0].float()
            y_valence = test_data[1]
            y_energy = test_data[2]
            y_danceability = test_data[3]
            
        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        y_predict = model(X)                                    # get model predictions for batch
        
        if device:                                                 # covert to lists, depending on GPU or CPU 
            y_predict_valence= y_predict[:,0].data.cpu().numpy().tolist()
            y_predict_energy= y_predict[:,1].data.cpu().numpy().tolist()
            y_predict_danceability= y_predict[:,2].data.cpu().numpy().tolist()
            y_true_valence=y_valence.data.cpu().numpy().tolist()
            y_true_energy=y_energy.data.cpu().numpy().tolist()
            y_true_danceability=y_danceability.data.cpu().numpy().tolist()
            
        else:
            y_predict=y_predict.data.numpy().tolist()
            y=y.data.numpy().tolist()
        
        y_pred_valence+=y_predict_valence                                          # add predictions to predictions from previuos batches
        y_pred_energy+=y_predict_energy
        y_pred_danceability+=y_predict_danceability
        y_real_valence += y_true_valence                                                # add labels to labels from previous batches
        y_real_energy += y_true_energy
        y_real_danceability += y_true_danceability
        
    sp_v = stats.spearmanr(y_real_valence, y_pred_valence)[0]
    sp_e = stats.spearmanr(y_real_energy, y_pred_energy)[0]
    sp_d = stats.spearmanr(y_real_danceability, y_pred_danceability)[0]
    print("Spearman correlation valience:" , sp_v)                   # print evaluation metrics
    print("Spearman correlation energy:" ,sp_e)
    print("Spearman correlation danceability:",sp_d)
    print("Spearman correlation :" ,(sp_v +sp_e+sp_d)/3)
    return

#dataset split
batch_size = 32
val_size = 0.10
test_size=0.10

specs_multi_valence = SpectrogramDatasetMulti(
         '../input/patreco3-multitask-affective-music/data/multitask_dataset/',
         train=True)

train_loader_val,validation_loader_val, test_loader_val = torch_train_val_test_split(specs_multi_valence, batch_size ,batch_size, batch_size, val_size, test_size)

output_size = 10
num_epochs = 20
learning_rate = 1e-4
dropout = 0.05
momentum=0.99
weight_decay = 1e-4
early_stopping = True

model = CNN(output_size,momentum,dropout)                                                           # initialize model                                                                
criterion=Multitask_loss(nn.MSELoss().to(device),[0.4,0.3,0.3])                                     # initialize custom loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)       # optimizer with L2 normalizaton

model,train_losses,train_losses_valence, train_losses_energy, train_losses_danceability,validation_losses, validation_losses_valence, validation_losses_energy, validation_losses_danceability, epochs_run = train_CNNmodel_multi(model,optimizer,criterion,num_epochs,train_loader_val,validation_loader=validation_loader_val,validate=True, early_stopping=early_stopping,patience=3)

x=np.arange(epochs_run)
plt.plot(x,train_losses,label='Training loss')
plt.plot(x,validation_losses,label='Validation loss')
plt.title('CNN: Training and validation loss danceability per epoch for mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

x=np.arange(epochs_run)
plt.plot(x,train_losses_valence,label='Training loss valence')
plt.plot(x,validation_losses_valence,label='Validation loss valence')
plt.title('CNN: Training and validation loss danceability per epoch for mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

x=np.arange(epochs_run)
plt.plot(x,train_losses_energy,label='Training loss energy')
plt.plot(x,validation_losses_energy,label='Validation loss energy')
plt.title('CNN: Training and validation loss danceability per epoch for mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

x=np.arange(epochs_run)
plt.plot(x,train_losses_danceability,label='Training loss danceability')
plt.plot(x,validation_losses_danceability,label='Validation loss danceability')
plt.title('CNN: Training and validation loss danceability per epoch for mel spectrogram',fontsize=12)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# testing
evaluate_CNN_multi(model, test_loader_val ,batch_size)