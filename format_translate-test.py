# Read startphrases, ending1, ending2 and labels

import pandas as pd
from collections import defaultdict

languages = ["hi", "su", "id", "jv", "kn", "sw"]

ending1 = defaultdict(list)
ending2 = defaultdict(list)
labels = defaultdict(list)
startphrases = defaultdict(list)
for lang in languages:
    with open("translated_data/startphrase.{}_to_en.tsv".format(lang), 'r') as f:
        for line in f:
            startphrases[lang].append(line.strip().split('\t')[1].strip())
    with open("translated_data/ending1.{}_to_en.tsv".format(lang), 'r') as f:
        for line in f:
            ending1[lang].append(line.strip().split('\t')[1].strip())
    with open("translated_data/ending2.{}_to_en.tsv".format(lang), 'r') as f:
        for line in f:
            ending2[lang].append(line.strip().split('\t')[1].strip())
    with open("langdata/{}.csv".format(lang), 'r') as f:
        reader = pd.read_csv(f, delimiter=',', encoding='utf-8')
        labels[lang] = reader['labels'].tolist()
        for i,label in enumerate(labels[lang]):
            labels[lang][i] = int(label)
            

# assert len(startphrases) == len(ending1) == len(ending2) == len(labels)
for lang in languages:
    assert len(startphrases[lang]) == len(ending1[lang]) == len(ending2[lang]) == len(labels[lang])

# Pop elements from startphrases, ending1, ending2 and labels if len(string) == 0
for lang in languages:
    for i in range(len(startphrases[lang])-1, -1, -1):
        if len(startphrases[lang][i]) == 0 or len(ending1[lang][i]) == 0 or len(ending2[lang][i]) == 0:
            startphrases[lang].pop(i)
            ending1[lang].pop(i)
            ending2[lang].pop(i)
            labels[lang].pop(i)
    
# assert startphrases, ending1, ending2 are strings and labels are integers
for lang in languages:
    print("Checking {}...".format(lang))
    for i in range(len(startphrases[lang])):
        assert isinstance(startphrases[lang][i], str) and len(startphrases[lang][i]) > 0
        assert isinstance(ending1[lang][i], str) and len(ending1[lang][i]) > 0
        assert isinstance(ending2[lang][i], str) and len(ending2[lang][i]) > 0
        assert isinstance(labels[lang][i], int)

# Write startphrases, ending1, ending2 and labels to a csv file for each language using pandas
for lang in languages:
    data = {'startphrase': startphrases[lang], 'ending1': ending1[lang], 'ending2': ending2[lang], 'labels': labels[lang]}
    df = pd.DataFrame(data)
    df.to_csv("{}_{}.csv".format(lang, lang), index=False)


