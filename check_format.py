import pandas as pd
data = pd.read_csv("langdata/sw.csv")
data.head()

# Get startphrase, ending1, ending2 and labels
startphrase = data['startphrase'].tolist()
ending1 = data['ending1'].tolist()
ending2 = data['ending2'].tolist()
labels = data['labels'].tolist()

# assert len(startphrases) == len(ending1) == len(ending2) == len(labels)
assert len(startphrase) == len(ending1) == len(ending2) == len(labels)

# Pop elements from startphrases, ending1, ending2 and labels if len(string) == 0
for i in range(len(startphrase)-1, -1, -1):
    if not isinstance(startphrase[i], str) or not isinstance(ending1[i], str) or not isinstance(ending2[i], str) or not isinstance(labels[i], int):
        print(i)
    if len(startphrase[i]) == 0 or len(ending1[i]) == 0 or len(ending2[i]) == 0:
        startphrase.pop(i)
        ending1.pop(i)
        ending2.pop(i)
        labels.pop(i)
    
# assert startphrases, ending1, ending2 are strings and labels are integers
for i in range(len(startphrase)):
    assert isinstance(startphrase[i], str) and len(startphrase[i]) > 0
    assert isinstance(ending1[i], str) and len(ending1[i]) > 0
    assert isinstance(ending2[i], str) and len(ending2[i]) > 0
    assert isinstance(labels[i], int)

# Write startphrases, ending1, ending2 and labels to a csv file for each language using pandas
data = {'startphrase': startphrase, 'ending1': ending1, 'ending2': ending2, 'labels': labels}
df = pd.DataFrame(data)
df.to_csv("langdata/kn_new.csv", index=False)
    