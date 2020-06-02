'''
To run this file, two arguements are expected:
Command Example: python model_bilstm_trainable.py restaurant

Change the experimental setup values to match the experiment to be currently conducted

For any new domain, create folders inside dataset, models, ontology and logs folder
ref: https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/main.py

For attenion reference: https://blog.floydhub.com/attention-mechanism/
'''

import sys
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
from torch.nn import functional as F
import pickle
import numpy as np
import random
import json
import os
# from bert_embedding import BertEmbedding

# bert_embedding = BertEmbedding()

# experimental setup values
seed = 1234
batch_size = 100
domain = sys.argv[1]
sampling = "no"                             # down|up|no  -  just for logs
weighted = "weighted"                     # unweighted/weighted - just for logs
dataamount = "50000"

def seed_everything():
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

seed_everything()

print(weighted + " - " + sampling + " sampled - " + dataamount + " - " + str(seed))

torch.set_printoptions(edgeitems=2)
# torch.set_printoptions(profile="full")

train_sentences = pickle.load(open(f'dataset/' + domain + '/train_sentences.pkl', 'rb'))
val_sentences = pickle.load(open(f'dataset/' + domain + '/val_sentences.pkl', 'rb'))
test_sentences = pickle.load(open(f'dataset/' + domain + '/test_sentences.pkl', 'rb'))
train_labels = pickle.load(open(f'dataset/' + domain + '/train_labels.pkl', 'rb'))
val_labels = pickle.load(open(f'dataset/' + domain + '/val_labels.pkl', 'rb'))
test_labels = pickle.load(open(f'dataset/' + domain + '/test_labels.pkl', 'rb'))

train_data = TensorDataset(torch.from_numpy(train_sentences), torch.from_numpy(train_labels))
val_data = TensorDataset(torch.from_numpy(val_sentences), torch.from_numpy(val_labels))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))

train_loader = DataLoader(train_data, shuffle=False, batch_size=batch_size)
val_loader = DataLoader(val_data, shuffle=False, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda:2")
else:
    device = torch.device("cpu")

def softmax(l):
    return np.exp(l)/np.sum(np.exp(l)) 

# For Weighted Word Embeddings 
with open("./ontology/" + domain + "/scores.json", "r") as f:
    scores = json.load(f)

word2idx = pickle.load(open(f'dataset/' + domain + '/word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'dataset/' + domain + '/idx2word.pkl', 'rb'))
cn_words = pickle.load(open(f'embeddings/cn_nb.300_words.pkl', 'rb'))
cn_word2idx = pickle.load(open(f'embeddings/cn_nb.300_idx.pkl', 'rb'))
cn_embs = pickle.load(open(f'embeddings/cn_nb.300_embs.pkl', 'rb'))

# Bidirectional recurrent neural network (many-to-one)
class AttnBiLSTM(nn.Module):

    def __init__(self, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(AttnBiLSTM, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)

        # scores_matrix = torch.ones((vocab_size, 1))
        # weights_matrix = torch.ones((vocab_size, embedding_dim))
        # for v in target_vocab:
        #     ''' initialize weights_matrix with conceptnet embeddings '''
        #     try:
        #         if v in ['_PAD','_UNK']:
        #             weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[0])
        #         else:
        #             weights_matrix[word2idx[v]] = torch.from_numpy(cn_embs[cn_word2idx[v]])
        #     except:
        #         pass
            # if v in scores.keys():
            #     scores_matrix[word2idx[v], 0] = scores[v]

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding.weight = torch.nn.Parameter(weights_matrix)
        # self.embedding.weight.requires_grad = False
        # self.aspect_scores = nn.Embedding(vocab_size, 1)
        # self.aspect_scores.weight = torch.nn.Parameter(scores_matrix)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)                                # 2 for bidirection, when dropout
        self.sigmoid = nn.Sigmoid()
        # self.label = nn.Linear(hidden_dim*2, output_size)


    def attention_net(self, embeds, weights):
        '''
		Now we will incorporate Attention mechanism in our BiLSTM model. In this new model, we will use attention 
        to compute soft alignment score corresponding between each of the hidden_state and the last hidden_state 
        of the LSTM. We will be using torch.bmm for the batch matrix multiplication.
		
		Arguments
		---------
		
		lstm_output : Final output of the LSTM which contains hidden layer outputs for each sequence.
		final_state : Final time-step hidden state (h_n) of the LSTM
		
		---------
		
		Returns : It performs attention mechanism by first computing weights for each of the sequence present in lstm_output and and then finally computing the
				  new hidden state.
				  
		Tensor Size :
					hidden.size() = (batch_size, hidden_size)
					attn_weights.size() = (batch_size, num_seq)
					soft_attn_weights.size() = (batch_size, num_seq)
					new_hidden_state.size() = (batch_size, hidden_size)
					  
		'''
        # hidden = hidden.squeeze(0)
        # attn_weights = torch.bmm(embeds, hidden.unsqueeze(2)).squeeze(2)
        # soft_attn_weights = F.softmax(attn_weights)
        # return new_embeds

        weights = weights.squeeze(0)
        attn_weights = torch.bmm(embeds, weights.unsqueeze(2))
        soft_attn_weights = F.softmax(attn_weights).repeat(1, 1, embedding_dim)
        new_embeds = embeds*soft_attn_weights
        return new_embeds


    def forward(self, x, hidden):                                                       # x.shape =  torch.Size([10, 200]) (batch_size, seq_length)
        embeds = self.embedding(x)                                                      # embeds.shape =  torch.Size([10, 200, 100])
        # hidden = torch.ones(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # h0.shape =  torch.Size([2, 10, 512]) : 2 for bidirection
        weights = torch.ones(1, batch_size, embedding_dim).to(device)                      # weights.shape = torch.Size([1, 10, 300])

        ''' attention mechanism '''
        embeds = self.attention_net(embeds, weights)

        ''' Set initial states '''
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # h0.shape =  torch.Size([2, 10, 512]) : 2 for bidirection
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # c0.shape =  torch.Size([2, 10, 512])

        # h1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # same as h0 
        # c1 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)     # same as c0

        ''' Forward propagate the weighted/unweighted embeddings to Bi-LSTM '''
        lstm_out, (hidden, cell) = self.lstm(embeds, (h0, c0))                          # lstm_out.shape =  torch.Size([10, 200, 1024]) (batch_size, seq_length, hidden_size*2) | hidden.shape =  torch.Size([2, 10, 512]) | cell.shape =  torch.Size([2, 10, 512])
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)                   # after dropout: hidden.shape =  torch.Size([10, 1024])

        fc_out = self.fc(hidden)
        out = self.sigmoid(fc_out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device), weight.new(self.num_layers, batch_size, self.hidden_size).zero_().to(device))
        return hidden

vocab_size = len(word2idx) + 1
output_size = 1
seq_length = 300
embedding_dim = 300
hidden_dim = 512
n_layers = 1

target_vocab = word2idx.keys()

model = AttnBiLSTM(vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers).to(device)

lr=0.005
criterion = nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

epochs = 2
counter = 0
print_every = 10
clip = 5
valid_loss_min = np.Inf
train_loss_min = np.Inf

# initial_weights = model.aspect_scores.weight.squeeze(1).detach().cpu()                  # model.aspect_scores before training

print("Start training..")
model.train()

for i in range(epochs):
    h = model.init_hidden(batch_size)

    for inputs, labels in train_loader:
        counter += 1
        h = tuple([e.data for e in h])
        inputs, labels = inputs.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        model.zero_grad()
        output = model(inputs, h)
        # print("output")
        # print(output)
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
                torch.save(model.state_dict(), 'models/' + domain + '/e_attention_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt')
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                valid_loss_min = np.mean(val_losses)

# final_weights = model.aspect_scores.weight.squeeze(1).detach().cpu()                   # model.aspect_scores after training
# diff_weights = torch.abs(initial_weights - final_weights)

# f = open('logs/' + domain + '/attention_' + str(seed) + '_' + weighted + '_' + dataamount + '.csv', 'w+')
# f.write('term,initial,trained,diff')
# for v in target_vocab:
#     if v in scores.keys():
#         word_i = word2idx[v]
#         f.write('\n' + v + ',' + str(initial_weights[word_i].numpy()) + ',' + str(final_weights[word_i].numpy()) + ',' + str(diff_weights[word_i].numpy()))
# f.close()

# Loading the best model
model.load_state_dict(torch.load('models/' + domain + '/e_attention_' + weighted + '_' + dataamount + '_' + str(seed) + '.pt'))

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
print(domain + " - " + sampling + " sampled - " + dataamount + " - " + str(seed))
