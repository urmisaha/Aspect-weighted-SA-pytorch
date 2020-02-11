import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
import pickle
import numpy as np
import random
import json
import os

def seed_everything(seed=2341):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_everything()

print("cp weights - only embedding - positive downsampled - 2341")

# torch.set_printoptions(edgeitems=5)
torch.set_printoptions(profile="full")

train_sentences = pickle.load(open(f'train_sentences.pkl', 'rb'))
val_sentences = pickle.load(open(f'val_sentences.pkl', 'rb'))
test_sentences = pickle.load(open(f'test_sentences.pkl', 'rb'))
train_labels = pickle.load(open(f'train_labels.pkl', 'rb'))
val_labels = pickle.load(open(f'val_labels.pkl', 'rb'))
test_labels = pickle.load(open(f'test_labels.pkl', 'rb'))

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

batch_size = 100

train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def softmax(l):
    return np.exp(l)/np.sum(np.exp(l)) 

# For Weighted Word Embeddings 
aspect_term_list = pickle.load(open("aspect_term_list.pkl", "rb"))
aspect_weights = pickle.load(open("aspect_weights.pkl", "rb"))

for key, val in aspect_weights.items():
    aspect_weights[key] = np.around(val, decimals=1)

aspect_weights['ambience'] = 0.7

aspect_term_mapping = pickle.load(open("aspect_term_mapping.pkl", "rb"))

# with open("./ontology/restaurant/concepts_list.pkl", "rb") as f:
#     concepts_list = pickle.load(f)

# with open("./ontology/restaurant/scores.json", "r") as f:
#     scores = json.load(f)

word2idx = pickle.load(open(f'word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'idx2word.pkl', 'rb'))

# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)

        weights_matrix = torch.ones((vocab_size, 1))
        
        for v in target_vocab:
            word_i = word2idx[v]
            if int(word_i) != 0 and v in aspect_term_list:
                weights_matrix[word2idx[v], 0] = float(aspect_weights[aspect_term_mapping[v]]*10)

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.aspect_scores = nn.Embedding(vocab_size, 1)
        # self.aspect_scores2 = nn.Embedding(vocab_size, 1)
        self.aspect_scores.weight = torch.nn.Parameter(weights_matrix)
        # self.aspect_scores2.weight = torch.nn.Parameter(weights_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_dim*2, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim, output_size)            # 2 for bidirection
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):                               # x.shape =  torch.Size([10, 200]) (batch_size, seq_length)
        scores_matrix = self.aspect_scores(x)                   # scores_matrix.shape =  torch.Size([10, 200, 1])
        # scores_matrix2 = self.aspect_scores(x)                  # scores_matrix2.shape =  torch.Size([10, 200, 1])
        embeds = self.embedding(x)                              # embeds.shape =  torch.Size([10, 200, 100])
        # scores_matrix1 = scores_matrix.repeat(1, 1, embedding_dim)                    # scores_matrix1.shape =  torch.Size([10, 200, emb_dim])
        # embeds = embeds*scores_matrix1
        
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # h0.shape =  torch.Size([2, 10, 512]) : 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # c0.shape =  torch.Size([2, 10, 512])

        h1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # same as h0 
        c1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device) # same as c0
        
        # Forward propagate LSTM
        lstm_out, (hidden, cell) = self.lstm(embeds, (h0, c0))                      # lstm_out.shape =  torch.Size([10, 200, 1024]) (batch_size, seq_length, hidden_size*2) | hidden.shape =  torch.Size([2, 10, 512]) | cell.shape =  torch.Size([2, 10, 512])

        # Multiplying weights to outputs of first bilstm layer
        scores_matrix2 = scores_matrix.repeat(1, 1, hidden_dim*2)                    # scores_matrix2.shape =  torch.Size([10, 200, 1024])
        lstm_out = lstm_out*scores_matrix2
        lstm_out, (hidden, cell) = self.lstm2(lstm_out, (h1, c1))                   # lstm_out: tensor of shape (batch_size, seq_length, hidden_size*2)
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], cell[-1,:,:]), dim = 1))   # after dropout: hidden.shape =  torch.Size([10, 1024])
        # hidden = self.dropout(torch.cat((hidden[-2,:,:], cell[-1,:,:]), dim = 1))   # after dropout: hidden.shape =  torch.Size([10, 1024])
    
        # Decode the hidden state of the last time step
        lstm_out = self.fc(hidden[-2,:,:])
        lstm_out = self.sigmoid(lstm_out)
        return lstm_out
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device), weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 100
hidden_dim = 512
n_layers = 1

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

initial_weights = model.aspect_scores.weight.squeeze(1).detach().cpu()    # model.aspect_scores before training

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
        labels = labels.unsqueeze(1)
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
                lab = lab.unsqueeze(1)
                out = model(inp, val_h)
                val_loss = criterion(out, lab.float())
                val_losses.append(val_loss.item())
                
            model.train()
            print("Epoch: {}/{}...".format(i+1, epochs),
                  "Step: {}...".format(counter),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))
            if np.mean(val_losses) < valid_loss_min:
                torch.save(model.state_dict(), 'models/state_dict_val_cp_downsample2341.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

final_weights = model.aspect_scores.weight.squeeze(1).detach().cpu()                   # model.aspect_scores after training
diff_weights = torch.abs(initial_weights - final_weights)

trained_scores_dict = {}
f = open('cp_scores_trained_bilstm.csv', 'w+')
f.write('term,initial,trained,diff')
for v in target_vocab:
    if v in aspect_term_list:
        word_i = word2idx[v]
        f.write('\n' + v + ',' + str(initial_weights[word_i].numpy()) + ',' + str(final_weights[word_i].numpy()) + ',' + str(diff_weights[word_i].numpy()))
        trained_scores_dict[v] = {'initial': initial_weights[word_i], 'trained': final_weights[word_i], 'diff': diff_weights[word_i]}
f.close()
pickle.dump(trained_scores_dict, open(f'cp_scores_trained_dict_bilstm.pkl', 'wb'))
            
# Loading the best model
model.load_state_dict(torch.load('models/state_dict_val_cp_downsample2341.pt'))

test_losses = []
num_correct = 0
h = model.init_hidden(batch_size)

model.eval()

total_labels = torch.LongTensor()
total_preds = torch.LongTensor()

for inputs, labels in test_loader:
    h = tuple([each.data for each in h])
    inputs, labels = inputs.to(device), labels.to(device)
    labels = labels.unsqueeze(1)
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
print("num_layers=1 - cp weights - only embedding - positive downsampled - 2341")
