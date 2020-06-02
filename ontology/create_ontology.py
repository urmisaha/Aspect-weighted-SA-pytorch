'''
INSTRUCTIONS:

Arguements of this file:
domain
limit for extraction for each node
scores for roots
scores for aspects (aspect_categories - special nodes)

*** CREATE A FOLDER CALLED "[domain]" UNDER FOLDER "ontology" ***

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

# Dictionary holding domains(all domains on which experiments are being performed) mapped to their aspect categories
'''
This part can be taken as input from users: domain and its categories.
Create this dictionary on getting the inputs. Then proceed with the rest of the code.
'''
aspect_categories = {}
aspect_categories["restaurant"] = ["food", "service", "price", "ambience"]
aspect_categories["movie"] = ["music", "acting", "plot", "direction"]
aspect_categories["music"] = ["genre", "lyrics", "artist"]
aspect_categories["ride"] = ["driver", "fare", "time"]
aspect_categories["hotel"] = ["room", "cleanliness", "value", "service", "location", "checkin", "business"] 

root = sys.argv[1]
limit = sys.argv[2]
rt_score = float(sys.argv[3])
as_score = float(sys.argv[4])

folder = root

# these are the three data dumped in pickle files
concepts = []
score = {}
parent = {}

aspects = []
aspect_scores = {}

parent[root] = ""
aspects.append(root)
aspect_scores[root] = rt_score
for aspect in aspect_categories[root]:
    parent[aspect] = root
    aspects.append(aspect)
    aspect_scores[aspect] = as_score
print(aspects)

def extract_concepts(node, node_limit):
    synonyms = [node.lower()]
    for syn in wordnet.synsets(node):
        for l in syn.lemmas():
            if "_" not in l.name():
                synonyms.append(l.name().lower())
    synonyms = set(synonyms)
    for syn_p in synonyms:
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
                            # score[ch] = float(score[syn_p])/2       # weights of child nodes are half of those of parent nodes
                            # score[ch] = float(e['weight'])          # weights are taken from conceptnet property "weight"
                            score[ch] = float(score[syn_p])/2 + float(e['weight'])          # both conceptnet "weight" and half of parent's score
                            concepts.append(ch)
                            parent[ch] = syn_p
            except:
                pass
    if node == root:
        print("root:::::: " + node)
    else:
        print("node: " + node)
    print(len(synonyms))    
    print(synonyms)


for concept in aspects:
    concepts.append(concept)
    score[concept] = aspect_scores[concept]
    extract_concepts(concept, limit)

with open("./"+folder+"/scores.json", "w+") as f:
    f.write(json.dumps(score, indent=4, sort_keys=True))
f.close()

with open("./"+folder+"/parents.json", "w+") as f:
    f.write(json.dumps(parent, indent=4, sort_keys=True))
f.close()

pickle.dump(concepts, open('./'+folder+'/concepts_list.pkl', 'wb'))

# print("Done")