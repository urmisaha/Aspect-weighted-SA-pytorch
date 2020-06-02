import numpy as np
# import bcolz
import pickle

words = []
idx = 0
word2idx = {}
# vectors = bcolz.carray(np.zeros(1), rootdir=f'6B.100.dat', mode='w')
embeddings = []

with open(f'../../common_resources/glove.840B.300d.txt', 'rb') as f:
    for l in f:
        try:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            emb = np.array(line[1:]).astype(np.float)
            embeddings.append(emb)
        except:
            print(l)
            pass
        # vect = np.array(line[1:]).astype(np.float)
        # vectors.append(vect)
    
# vectors = bcolz.carray(vectors[1:].reshape((400000, 100)), rootdir=f'6B.100.dat', mode='w')
# vectors.flush()
pickle.dump(words, open(f'embeddings/840B.300_words.pkl', 'wb'))
pickle.dump(word2idx, open(f'embeddings/840B.300_idx.pkl', 'wb'))
pickle.dump(embeddings, open(f'embeddings/840B.300_embs.pkl', 'wb'))