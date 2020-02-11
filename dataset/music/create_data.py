import json

f = open("./Digital_Music_5.json", "r")
f_train = open("./train.csv", "w+")
f_test = open("./test.csv", "w+")

c = 0
for line in f.readlines():
    item = json.loads(line)
    item['reviewText'] = item['reviewText'].replace(","," ")
    c = c + 1
    if c <= 48000:
        f_train.write(item['reviewText'])
        if item ['overall'] <= 2.0:
            f_train.write(",0\n")
        else:
            f_train.write(",1\n")
    else:
        f_test.write(item['reviewText'])
        if item ['overall'] <= 2.0:
            f_test.write(",0\n")
        else:
            f_test.write(",1\n")