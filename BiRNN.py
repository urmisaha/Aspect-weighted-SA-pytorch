import torch
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
# from preprocessing import *
import bcolz
import pickle
import numpy as np

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

word2idx = pickle.load(open(f'word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'idx2word.pkl', 'rb'))

with open("aspect_term_list.pkl", "rb") as f:
    aspect_term_list = pickle.load(f)

with open("aspect_weights.pkl", "rb") as f:
    aspect_weights = pickle.load(f)

with open("aspect_term_mapping.pkl", "rb") as f:
    aspect_term_mapping = pickle.load(f)


# Bidirectional recurrent neural network (many-to-one)
class BiRNN(nn.Module):

    def __init__(self, vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_dim
        self.num_layers = n_layers
        self.dropout = nn.Dropout(drop_prob)

        # Pretrained embedding multiplied by aspect weights
        vectors = bcolz.open(f'6B.100.dat')[:]
        words = pickle.load(open(f'6B.100_words.pkl', 'rb'))
        word2idx = pickle.load(open(f'6B.100_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]-1] for w in words}

        self.glove = glove

        matrix_len = len(target_vocab)
        weights_matrix = np.zeros((matrix_len, 100))
        words_found = 0

        for i, word in enumerate(target_vocab):
            try:
                if word in aspect_term_list: 
                    weights_matrix[i] = glove[word] * (1 - aspect_weights[aspect_term_mapping[word]])
                words_found += 1
            except KeyError:
                weights_matrix[i] = np.random.normal(scale=0.6, size=(embedding_dim, ))

        # self.embedding, embedding_dim = create_emb_layer(weights_matrix, True)
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, output_size)  # 2 for bidirection
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):
        embeds = self.embedding(x)

        avg_aspect_W = []
        for s, sent in enumerate(x):
            for i, e in enumerate(sent):
                weights_s = []
                if int(e) != 0 and idx2word[int(e)] in aspect_term_list:
                    word = idx2word[int(e)]
                    w = aspect_weights[aspect_term_mapping[word]]
                    # Multiplying embedding layer outputs with aspect weights
                    embeds[s][i] *= w

                    # Creating weight matrix to multiply with the outputs of last hidden layer
                    weights_s.append(w)
            a = 10*np.mean(weights_s) if len(weights_s) > 0 else 1
            avg_aspect_W.append([a]*1024)
        
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