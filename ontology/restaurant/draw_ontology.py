import json

with open("parents_new.json", "r") as f:
    concepts = json.load(f)

level_one = []
level_two = []
level_three = []


for key, val in concepts.items():
    if val == "":
        level_one.append(key)

for item in level_one:
    print(item)

print("\t|")
print("\t|")
print("\t->")
for key, val in concepts.items():
    if val in level_one:
        level_two.append(key)

for item in level_two:
    print("\t\t"+item)


for key, val in concepts.items():
    if val in level_two:
        level_three.append(key)

for item in level_three:
    print("\t\t\t\t"+item)