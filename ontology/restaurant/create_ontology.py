# arguements of this file:
'''
domain
limit for extraction for each node
scores for restaurant
scores for food
scores for service
scores for price
scores for ambience
'''

import requests
import sys
import json
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from nltk.corpus import wordnet

stopwords = set(stopwords.words('english'))
ps = PorterStemmer()

aspect_scores = {}
root = sys.argv[1]
limit = sys.argv[2]
aspect_scores['restaurant'] = float(sys.argv[3])
aspect_scores['food'] = float(sys.argv[4])
aspect_scores['service'] = float(sys.argv[5])
aspect_scores['price'] = float(sys.argv[6])
aspect_scores['ambience'] = float(sys.argv[7])

# for i in range(1, 8):
#     print(sys.argv[i])


concepts = []
score = {}
parent = {}

def extract_concepts(node, node_limit):
    synonyms = [node.lower()]
    for syn in wordnet.synsets(node):
        for l in syn.lemmas():
            if "_" not in l.name():
                synonyms.append(l.name().lower())
    synonyms = set(synonyms)
    for syn_p in synonyms:
        if node == "restaurant":
            parent[syn_p] = ""
        score[syn_p] = score[node]
        tree = requests.get('http://api.conceptnet.io/c/en/'+syn_p+'?offset=0&limit='+node_limit).json()
        for e in tree['edges']:
            try:
                if e['end']['@type'] == "Node" and e['start']['@type'] == "Node" and e['start']['language'] == "en" and e['end']['language'] == "en":
                    child = ""                                                               
                    if e['start']['label'] != syn_p:
                        child = e['start']['label']
                    else:
                        child = e['end']['label']
                    for ch in child.split():
                        ch = ch.lower()
                        if ch not in stopwords and ch not in ['a', 'an', 'the'] and ps.stem(syn_p) != ps.stem(ch) and ch not in concepts:
                            # concepts.append({'start': syn_p, 'end': ch, 'weight': e['weight']})
                            score[ch] = float(score[syn_p])/2
                            concepts.append(ch)
                            parent[ch] = syn_p
            except:
                pass
    print("node: " + node)
    print(synonyms)
    print(len(synonyms))    


for concept in ["food", "service", "price", "ambience", "restaurant"]:
    if concept != root:
        parent[concept] = root
    concepts.append(concept)
    score[concept] = aspect_scores[concept]
    extract_concepts(concept, limit)

with open("scores_new.json", "w+") as f:
    f.write(json.dumps(score, indent=4, sort_keys=True))
f.close()

with open("parents_new.json", "w+") as f:
    f.write(json.dumps(parent, indent=4, sort_keys=True))
f.close()

pickle.dump(concepts, open(f'concepts_list_new.pkl', 'wb'))

# print("Done")