import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import precision_recall_fscore_support
import torch.nn as nn
# from preprocessing import *
import bcolz
import pickle
import numpy as np
from BiRNN import *

batch_size = 400

test_sentences = pickle.load(open(f'test_sentences.pkl', 'rb'))
test_labels = pickle.load(open(f'test_labels.pkl', 'rb'))
test_data = TensorDataset(torch.from_numpy(test_sentences), torch.from_numpy(test_labels))
test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

# torch.cuda.is_available() checks and returns a Boolean True if a GPU is available, else it'll return False
is_cuda = torch.cuda.is_available()

# If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
if is_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

word2idx = pickle.load(open(f'word2idx.pkl', 'rb'))
idx2word = pickle.load(open(f'idx2word.pkl', 'rb'))

vocab_size = len(word2idx) + 1
output_size = 1
embedding_dim = 100
hidden_dim = 512
n_layers = 2

target_vocab = word2idx.keys()

model = BiRNN(vocab_size, target_vocab, output_size, embedding_dim, hidden_dim, n_layers).to(device)

model.load_state_dict(torch.load('./models/state_dict_val_loss_only_emb_pos_downsample.pt'))
model.eval()

h = model.init_hidden(batch_size)
criterion = nn.BCELoss().to(device)

test_losses = []
num_correct = 0

total_labels = torch.LongTensor()
total_preds = torch.LongTensor()

count = 0

for inputs, labels in test_loader:
    print(count)
    count += 1
    
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
