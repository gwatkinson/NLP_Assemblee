# -*- coding: utf-8 -*-

# In this section, we train our own model with gensim

import pandas as pd
from spacy.pipeline.tok2vec import DEFAULT_TOK2VEC_MODEL
import spacy

# First, run command: python -m spacy download fr_core_news_md 

import fr_core_news_md

sp = spacy.load("fr_core_news_md")
df = pd.read_csv("depinter_collapsed.csv")

df['interventions_spacy'] = df['interventions'].astype(str).apply(lambda x: sp(x))
df["interventions_spacy"] = df['interventions_spacy'].apply(
    lambda tokens: [token.lemma_ for token in tokens]
)
stop_words = sp.Defaults.stop_words | {"'", ",", ";", ":", " ", "", "."}

df["interventions_spacy"] = df['interventions_spacy'].apply(
    lambda words: [word for word in words if not word in stop_words]
)

# df.to_csv(r"mypath" + "\df_spacy.csv", index=False)

df_spacy = pd.read_csv("dataframes/df_spacy.csv")

texts = []
for intervention in list(df.interventions):
    texts.append(intervention)
len(texts)

nlp = fr_core_news_md.load()

config = {"model": DEFAULT_TOK2VEC_MODEL}
# nlp.add_pipe("tok2vec", config=config)

help(nlp)

doc = nlp("J'ai pris l'avion hier")
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)

texttry = texts[1]+texts[2]
texttry

# tovec = Tok2Vec(nlp, )