# Aspect-weighted sentiment analysis using domain ontology and deep neural network

We calculate aspect scores using two approaches:
1. conditional probability from dataset
2. a domain ontology

We incorporate these scores into our neural architecture to find the sentiment of a textual review. The scores are used to initialize a trainable layer of the neural architecture.

> **Domain:** Restaurant, Movie, Music, Uber Rides

### Steps to run
1. Run glove_embeddings.py if corresponding pickle files are already not created
2. Run preprocessing.py if corresponding pickle files are already not created
3. Run model.py

P.S: Read the commented introduction in each of the files mentioned to run the commands correctly.