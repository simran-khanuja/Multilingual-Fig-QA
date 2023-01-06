import pandas as pd
import pdb
import nltk
nltk.download("wordnet")
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from sklearn.metrics import jaccard_score
import pdb

lemmatizer = WordNetLemmatizer()
#stopwords = set(stopwords.words("english"))
determiners = ["the", "a", "an", "that"]

sns.set(font_scale=1.5)

def lemmatize_words(words):
    words = words.lower().translate(str.maketrans("", "", string.punctuation))

    return " ".join([lemmatizer.lemmatize(w) for w in word_tokenize(words) if w not in determiners])

def make_count_barplot(df: pd.DataFrame, out_filename: str, x_label: str, palette_name: str = "flare") -> None:
    plt.gca().clear()
    plt.figure(figsize=(8,4))
    ax = sns.countplot(data=df, x="subj", order=df["subj"].value_counts().index, palette=palette_name)
    plt.xlim(-0.5, 24)
    plt.xticks(rotation=90)
    plt.xlabel("")
    plt.ylabel("Count")
    plt.title(x_label)
    plt.tight_layout()
    plt.savefig(f"{out_filename}.png")
    plt.savefig(f"{out_filename}.eps")
    plt.close()

def get_hypernyms(sent: str, wordnet_pos: List) -> List:
    hypernyms = []
    #print(sent)
    for word, pos in zip(sent.split(), wordnet_pos):
        ignore = ["s", "at", "m", "in"]
        if word in ignore or (len(word) == 1 and word != "i"):
            continue
        # common errors
        if word == "he":
            word = "man"
        if word == "she":
            word = "woman"
        if word == "i":
            word = "me"
            
        if pos != "other":
            synset = wordnet.synsets(word, pos=pos)
        else:
            synset = wordnet.synsets(word)
        #for ss in synset:
            #print(word)
            #print(ss)
            #print(ss.hypernyms())
        if len(synset) != 0 and len(synset[0].hypernyms()) != 0:
            hypernym_names = [ss.name() for ss in synset[0].hypernyms()]
            hypernyms.extend(hypernym_names)
    #print(hypernyms)
    return hypernyms
    
def get_pos_tags(sent: str) -> List:
    return " ".join([x[1] for x in nltk.pos_tag(word_tokenize(sent))])

def get_wordnet_pos(treebank_tags: str) -> List:
    wordnet_tags = []
    for treebank_tag in treebank_tags.split():
        if treebank_tag.startswith('J'):
            wordnet_tags.append(wordnet.ADJ)
        elif treebank_tag.startswith('V'):
            wordnet_tags.append(wordnet.VERB)
        elif treebank_tag.startswith('N'):
            wordnet_tags.append(wordnet.NOUN)
        elif treebank_tag.startswith('R'):
            wordnet_tags.append(wordnet.ADV)
        else:
            wordnet_tags.append("other")
    return wordnet_tags

def correctness_by_pos(df: pd.DataFrame, threshold: int = 10) -> pd.DataFrame:
    counts = df["pos_tag"].value_counts()
    df = df[df['pos_tag'].isin(counts[counts >= threshold].index)]
    tag_correctness = {}
    for tag in df["pos_tag"].unique():
        tag_correctness[tag] = len(df.loc[(df["pos_tag"] == tag) & (df["correctness"] == True)])/len(df.loc[df["pos_tag"] == tag])

    return tag_correctness, dict(counts)

def correctness_by_hypernym(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    a = pd.Series([item for sublist in df["hypernyms"] for item in sublist])
    counts = dict(a.value_counts())
    selection = {key: item for key, item in counts.items() if item >= 10}
    mask = df["hypernyms"].apply(lambda x: any(item for item in selection if item in x))
    df = df[mask]

    hyp_correctness = {}
    for hypernym in selection:
        selected_rows = df[df["hypernyms"].apply(lambda x: hypernym in x)]
        hyp_correctness[hypernym] = len(selected_rows.loc[selected_rows["correctness"] == True])/len(selected_rows)
    
    return hyp_correctness, counts

if __name__ == "__main__":
    langs = ["hi", "id", "jv", "kn", "su", "sw", "en"]
    objects = {lang: [] for lang in langs}
    
    for lang in langs:
        segmented_file = f"data/syntax_chunked/syntax_tagged_{lang}.csv"
        #errors_file = "./gpt3_incorrect.csv"
        df = pd.read_csv(segmented_file, on_bad_lines="skip")
        df = df.dropna()
        #errors = set(list(pd.read_csv(errors_file)["startphrase"]))

        subj = df["x"].apply(lemmatize_words)
        subj = subj.loc[subj.shift() != subj] # only count subj, obj etc once per pair
        dummy_sub = pd.DataFrame({"subj": subj, "startphrase": df["startphrase"]})

        rel = df["y"].apply(lemmatize_words)
        rel = rel.loc[rel.shift() != rel]
        rel = rel.str.replace("wa\\b", "was", regex=True)
        rel = rel.str.replace("a\\b", "as", regex=True)
        rel = rel.str.replace("ha\\b", "has", regex=True)
        dummy_rel = pd.DataFrame({"subj": rel, "startphrase": df["startphrase"]})

        obj = df["z"].apply(lemmatize_words)
        obj = obj.loc[obj.shift() != obj]
        dummy_obj = pd.DataFrame({"subj": obj, "startphrase": df["startphrase"]})
        objects[lang].extend(dummy_obj["subj"].unique().tolist())

        subj_unique = len(subj.value_counts())
        rel_unique = len(rel.value_counts())
        obj_unique = len(obj.value_counts())

        make_count_barplot(dummy_sub, f"figures/subj_bar_{lang}", "Subject", "flare")
        make_count_barplot(dummy_rel, f"figures/rel_bar_{lang}", "Relation", "crest")
        make_count_barplot(dummy_obj, f"figures/obj_bar_{lang}", "Object", "viridis")
        print(f"lang: {lang}")
        print("unique subjects: ", subj_unique)
        print("unique relations: ", rel_unique)
        print("unique objects: ", obj_unique)

    # get the intersection of all objects
    shared_objects = set.intersection(*map(set, objects.values()))
    print("objects shared by all languages: ", len(shared_objects))
    print(shared_objects)

    # unique objects for each language
    for lang in langs:
        unique_objs = set(objects[lang]) - set.union(*[set(objects[other]) for other in langs if other != lang])
        print(f"unique objects for {lang}: ", len(unique_objs))
        print(unique_objs)

    # generate table for objects shared between 2 languages
    objs_intersections = {}
    jaccard_score = {}
    for lang1 in langs:
        for lang2 in langs:
            if lang1 != lang2:
                shared_objs = set(objects[lang1]) & set(objects[lang2])
                union_objs = set(objects[lang1]) | set(objects[lang2])
                j_s = len(shared_objs)/len(union_objs)
                jaccard_score[(lang1, lang2)] = j_s
    print(objs_intersections)
    print(jaccard_score)





