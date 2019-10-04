import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
# from preprocessing import *
# import bcolz
import pickle
import numpy as np
import random
import json
import os

def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_everything()

print("ontology weights - only hidden - negative upsampled - 1234")

torch.set_printoptions(edgeitems=10)
# torch.set_printoptions(profile="full")

train_sentences = pickle.load(open(f'train_sentences.pkl', 'rb'))
val_sentences = pickle.load(open(f'val_sentences.pkl', 'rb'))
test_sentences = pickle.load(open(f'test_sentences.pkl', 'rb'))
train_labels = pickle.load(open(f'train_labels.pkl', 'rb'))
val_labels = pickle.load(open(f'val_labels.pkl', 'rb'))
test_labels = pickle.load(open(f'test_labels.pkl', 'rb'))

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

batch_size = 200

train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# def create_emb_layer(weights_matrix, non_trainable=False):
#     # num_embeddings, embedding_dim = weights_matrix.size()
#     num_embeddings, embedding_dim = weights_matrix.shape
#     # emb_layer = nn.Embedding.from_pretrained(weights_matrix)
#     emb_layer = nn.Embedding(num_embeddings, embedding_dim)
#     emb_layer.weight.data.copy_(torch.from_numpy(weights_matrix))
#     # emb_layer.load_state_dict({'weight': weights_matrix})
#     if non_trainable:
#         emb_layer.weight.requires_grad = False

#     # return emb_layer, num_embeddings, embedding_dim
#     return emb_layer, embedding_dim

def softmax(l):
    return np.exp(l)/np.sum(np.exp(l)) 

# For Weighted Word Embeddings 
with open("aspect_term_list.pkl", "rb") as f:
    aspect_term_list = pickle.load(f)

with open("aspect_weights.pkl", "rb") as f:
    aspect_weights = pickle.load(f)

with open("aspect_term_mapping.pkl", "rb") as f:
    aspect_term_mapping = pickle.load(f)

with open("./ontology/restaurant/concepts_list_new.pkl", "rb") as f:
    concepts_list = pickle.load(f)

with open("./ontology/restaurant/scores_new.json", "r") as f:
    scores = json.load(f)

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)

        # Pretrained embedding multiplied by aspect weights
        # vectors = bcolz.open(f'6B.100.dat')[:]
        words = pickle.load(open(f'6B.100_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'6B.100_idx.pkl', 'rb'))

        # glove = {w: vectors[word2idx[w]-1] for w in words}

        # self.glove = glove

        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, 100))
        words_found = 0

        # for i, word in enumerate(target_vocab):
        #     try:
        #         if word in concepts_list: 
        #             weights_matrix[i] = glove[word] * aspect_weights[aspect_term_mapping[word]] * 10
        #         words_found += 1
        #     except KeyError:
        #         weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

        # self.embedding, embedding_dim = create_emb_layer(weights_matrix, True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)
        avg_aspect_W = []
        # count_one = 0
        # count_o = 0
        for s, sent in enumerate(x):
            # print(sent)     # 200 length vector of integers which are indices of words
            # r = random.random() * 10
            weights_s = []
            for i, e in enumerate(sent):
                if int(e) != 0 and idx2word[int(e)] in concepts_list:
                    word = idx2word[int(e)]
                    w = scores[word]
                    # Multiplying embedding layer outputs with aspect weights
                    # embeds[s][i] *= w*10
                    # embeds[s][i] *= r

                    # Creating weight matrix to multiply with the outputs of last hidden layer
                    weights_s.append(w)
            # a = 10*np.mean(weights_s) if len(weights_s) > 0 else 1    # uncomment this line for CP-weighted model
            a = np.mean(weights_s) if len(weights_s) > 0 else 1         # uncomment this line for ontology-weighted model        
            # a = r                                                     # uncomment this line for random-weighted model
            # a = 1                                                     # uncomment this line for unweighted hidden layer model
            # print(a)
            # if a == 1:
            #     count_one += 1
            # else:
            #     count_o += 1
            avg_aspect_W.append([a]*1024)
        # print("counts:")
        # print(count_one)
        # print(count_o)
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        
        # Forward propagate LSTM
        lstm_out, (hidden, cell) = self.lstm(embeds, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        
        # Multiplying weights to outputs of last hidden layer
        avg_aspect_W = torch.FloatTensor(avg_aspect_W).to(device)
        hidden = torch.mul(hidden, avg_aspect_W)

        # Decode the hidden state of the last time step
        lstm_out = self.fc(hidden)
        lstm_out = self.sigmoid(lstm_out)
        return lstm_out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device), weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

word2idx = pickle.load(open(f'word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'idx2word.pkl', 'rb'))

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 100
hidden_dim = 512
n_layers = 2

target_vocab = word2idx.keys()

model = BiRNN(vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers).to(device)

lr=0.005
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 10
clip = 5
valid_loss_min = np.Inf
train_loss_min = np.Inf
print("Start training..")
model.train()

for i in range(epochs):
    h = model.init_hidden(batch_size)
    # Checking whether batch contains a mixture of positive and negative samples
    # for inputs, labels in train_loader:
    #     print("labels:")
    #     print(labels)
    #     exit()

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        output = model(inputs, h)
        loss = criterion(output, labels.float())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        
        if counter%print_every == 0:
            val_h = model.init_hidden(batch_size)
            val_losses = []
            model.eval()
            for inp, lab in val_loader:
                val_h = tuple([each.data for each in val_h])
                inp, lab = inp.to(device), lab.to(device)
                out = model(inp, val_h)
                val_loss = criterion(out, lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) < valid_loss_min:
                torch.save(model.state_dict(), 'models/state_dict_val_onto_upsample2.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)


# Loading the best model
model.load_state_dict(torch.load('models/state_dict_val_onto_upsample2.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()

total_labels = torch.LongTensor()
total_preds = torch.LongTensor()

for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    output = model(inputs, h)
    test_loss = criterion(output, labels.float())
    test_losses.append(test_loss.item())
    pred = torch.round(output.squeeze())  # Rounds the output to 0/1
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)
    labels = labels.to("cpu").data.numpy()
    pred = pred.to("cpu").data.numpy()
    total_labels = torch.cat((total_labels, torch.LongTensor(labels)))
    total_preds = torch.cat((total_preds, torch.LongTensor(pred)))

print("Printing results::: ")
print(pred)
labels = total_labels.data.numpy()
preds = total_preds.data.numpy()

print("weighted precision_recall_fscore_support:")
print(precision_recall_fscore_support(labels, preds, average='weighted'))
print("============================================")

print(precision_recall_fscore_support(labels, preds, average=None))
print("============================================")
    
print("Test loss: {:.3f}".format(np.mean(test_losses)))
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}%".format(test_acc*100))
print("xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")
print("ontology weights - only hidden - upsampled - 1234")
