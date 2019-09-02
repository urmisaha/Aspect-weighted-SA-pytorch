# arguements of this file:
'''
root_score
aspect_score
number of concepts for restaurant
number of concepts for food
number of concepts for service
number of concepts for price
number of concepts for ambience
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

limits = {}
root_score = float(sys.argv[1])
aspect_score = float(sys.argv[2])
limits['restaurant'] = sys.argv[3]
limits['food'] = sys.argv[4]
limits['service'] = sys.argv[5]
limits['price'] = sys.argv[6]
limits['ambience'] = sys.argv[7]

root = "restaurant"

concepts = []
concepts.append(root)
score = {}
parent = {}
score[root] = root_score

def extract_concepts(root, root_limit):
    synonyms = []
    for syn in wordnet.synsets(root):
        for l in syn.lemmas():
            synonyms.append(l.name())
    synonyms = set(synonyms)
    print(synonyms)
    parents = []
    parents.append(root)
    for syno in synonyms:
        syno = syno.lower()
        if syno != root:
            score[syno] = score[root]
            parents.append(syno)
    for p in parents:
        tree = requests.get('http://api.conceptnet.io/c/en/'+p+'?offset=0&limit='+root_limit).json()
        # nodes=[]
        for e in tree['edges']:
            try:
                if e['end']['@type'] == "Node" and e['start']['@type'] == "Node" and e['start']['language'] == "en" and e['end']['language'] == "en":
                    child = ""                                                               
                    if e['start']['label'] != p:
                        child = e['start']['label']
                    else:
                        child = e['end']['label']
                    for ch in child.split():
                        ch = ch.lower()
                        if ch not in stopwords and ch not in ['a', 'an', 'the'] and ps.stem(p) != ps.stem(ch) and ch not in concepts:
                            # nodes.append({'start': p, 'end': ch, 'weight': e['weight']})
                            concepts.append(ch)
                            score[ch] = float(score[p])/2
                            parent[ch] = p
            except:
                pass


for concept in ["food", "service", "price", "ambience", "restaurant"]:
    if concept != "restaurant":
        concepts.append(concept)
        score[concept] = aspect_score
        parent[concept] = root
    extract_concepts(concept, limits[concept])

# print("--------")

with open("scores.json", "w+") as f:
    f.write(json.dumps(score, indent=4, sort_keys=True))
f.close()

with open("parents.json", "w+") as f:
    f.write(json.dumps(parent, indent=4, sort_keys=True))
f.close()

pickle.dump(concepts, open(f'concepts_list.pkl', 'wb'))

print("restaurant")
# print(key if key=='restaurant' for key, val in parent.items())

# for syn in synonyms:
#     print(syn)
