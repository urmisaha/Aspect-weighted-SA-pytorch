import bz2
from collections import Counter
import re
import nltk
import numpy as np
import pandas
import pickle
# nltk.download('punkt')

# train_file = bz2.BZ2File('train.ft.txt.bz2')
# test_file = bz2.BZ2File('test.ft.txt.bz2')
# print("bz2 completed..")

# train_file = train_file.readlines()
# test_file = test_file.readlines()
# print("readlines completed..")

# num_train = 800000 # We're training on the first 800,000 reviews in the dataset
# num_test = 200000 # Using 200,000 reviews from test set

# train_file = [x.decode('utf-8') for x in train_file[:num_train]]
# test_file = [x.decode('utf-8') for x in test_file[:num_test]]

dataframe = pandas.read_csv("merge_train.csv", header=None, names=['sentence', 'sentiment'])
dataset = dataframe.values
train_sentences = dataset[0:67600,0]
train_labels = dataset[0:67600,1].astype(int)

dataframe = pandas.read_csv("merge_test.csv", header=None, names=['sentence', 'sentiment'])
dataset = dataframe.values
test_sentences = dataset[0:16000,0]
test_labels = dataset[0:16000,1].astype(int)

print("Data load completed..")

# Extracting labels from sentences
# train_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in train_file]
# train_sentences = [x.split(' ', 1)[1][:-1].lower() for x in train_file]

# test_labels = [0 if x.split(' ')[0] == '__label__1' else 1 for x in test_file]
# test_sentences = [x.split(' ', 1)[1][:-1].lower() for x in test_file]

# Some simple cleaning of data
for i in range(len(train_sentences)):
    train_sentences[i] = re.sub('\d','0',train_sentences[i])

for i in range(len(test_sentences)):
    test_sentences[i] = re.sub('\d','0',test_sentences[i])

# Modify URLs to <url>
for i in range(len(train_sentences)):
    if 'www.' in train_sentences[i] or 'http:' in train_sentences[i] or 'https:' in train_sentences[i] or '.com' in train_sentences[i]:
        train_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", train_sentences[i])

for i in range(len(test_sentences)):
    if 'www.' in test_sentences[i] or 'http:' in test_sentences[i] or 'https:' in test_sentences[i] or '.com' in test_sentences[i]:
        test_sentences[i] = re.sub(r"([^ ]+(?<=\.[a-z]{3}))", "<url>", test_sentences[i])

words = Counter()  # Dictionary that will map a word to the number of times it appeared in all the training sentences

for i, sentence in enumerate(train_sentences):
    # The sentences will be stored as a list of words/tokens
    train_sentences[i] = []
    for word in nltk.word_tokenize(sentence):  # Tokenizing the words
        words.update([word.lower()])  # Converting all the words to lowercase
        train_sentences[i].append(word)
    if i%20000 == 0:
        print(str((i*100)/70000) + "% done")
print("100% done")

# Removing the words that only appear once
words = {k:v for k,v in words.items() if v>1}

# Sorting the words according to the number of appearances, with the most common word being first
words = sorted(words, key=words.get, reverse=True)

# Adding padding and unknown to our vocabulary so that they will be assigned an index
words = ['_PAD','_UNK'] + words

# Dictionaries to store the word to index mappings and vice versa
word2idx = {o:i for i,o in enumerate(words)}
idx2word = {i:o for i,o in enumerate(words)}

for i, sentence in enumerate(train_sentences):
    # Looking up the mapping dictionary and assigning the index to the respective words
    train_sentences[i] = [word2idx[word] if word in word2idx else 0 for word in sentence]

for i, sentence in enumerate(test_sentences):
    # For test sentences, we have to tokenize the sentences as well
    test_sentences[i] = [word2idx[word.lower()] if word.lower() in word2idx else 0 for word in nltk.word_tokenize(sentence)]

# Defining a function that either shortens sentences or pads sentences with 0 to a fixed length
def pad_input(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

seq_len = 200  # The length that the sentences will be padded/shortened to

train_sentences = pad_input(train_sentences, seq_len)
test_sentences = pad_input(test_sentences, seq_len)

# Converting our labels into numpy arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

split_frac = 0.5 # 50% validation, 50% test
split_id = int(split_frac * len(test_sentences))
val_sentences, test_sentences = test_sentences[:split_id], test_sentences[split_id:]
val_labels, test_labels = test_labels[:split_id], test_labels[split_id:]

pickle.dump(train_sentences, open(f'train_sentences.pkl', 'wb'))
pickle.dump(val_sentences, open(f'val_sentences.pkl', 'wb'))
pickle.dump(test_sentences, open(f'test_sentences.pkl', 'wb'))
pickle.dump(train_labels, open(f'train_labels.pkl', 'wb'))
pickle.dump(val_labels, open(f'val_labels.pkl', 'wb'))
pickle.dump(test_labels, open(f'test_labels.pkl', 'wb'))

pickle.dump(word2idx, open(f'word2idx.pkl', 'wb'))
pickle.dump(idx2word, open(f'idx2word.pkl', 'wb'))
