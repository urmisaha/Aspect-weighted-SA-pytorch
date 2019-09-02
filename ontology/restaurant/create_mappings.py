import json

f = open("rest.json", "r")
sents = f.readlines()
sents = json.loads(sents[0])
print(sents[3])

food = set([])
service = set([])
ambience = set([])
price = set([]) 

def add_map(ac, at):
    if ac == "food":
        food.add(at)
    elif ac == "service":
        service.add(at)
    elif ac == "ambience":
        ambience.add(at)
    elif ac == "price":
        price.add(at)


def find_terms(s):
    at = []
    try:
        at.append(s['aspectTerms']['aspectTerm']['@term'])
    except:
        try:
            for aterm in s['aspectTerms']['aspectTerm']:
                at.append(aterm['@term'])
        except:
            pass
    return at

def find_categories(s):
    ac = []
    try:
        ac.append(s['aspectCategories']['aspectCategory']['@category'])
    except:
        pass
    return ac

# def find_terms(s):
#     at = []
#     try:
#         at.append(s['aspectTerms']['aspectTerm']['@term'])
#     except:
#         try:
#             for aterm in s['aspectTerms']['aspectTerm']:
#                 at.append(aterm['@term'])
#         except:
#             pass
#     return at

# def find_categories(s):
#     ac = []
#     try:
#         ac.append(s['aspectCategories']['aspectCategory']['@category'])
#     except:
#         for acat in s['aspectCategories']['aspectCategory']:
#             ac.append(acat['@category'])
#     return ac


for s in sents:
    print(type(s))
    at = find_terms(s)
    ac = find_categories(s)
    for t in at:
        for c in ac:
            if c != 'anecdotes/miscellaneous':
                add_map(c, t)

print("food")
print(food)
print("service")
print(service)
print("ambience")
print(ambience)
print("price")
print(price)
