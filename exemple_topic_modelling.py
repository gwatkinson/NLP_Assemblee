# This is a code to create the topic modelling.
# This module needs the package and an enviromnent to be installed. To install the package, run the following command:
#       $ pip install -e .
# This works with the dataframe in ./data/processed/15th_merged_data_short.pkl

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from torch import nn

from nlp_assemblee.simple_trainer import LitModel, load_embedding

RESULT_DIR = Path("./results/topic_modelling/")

if __name__ == "__main__":
    print("Loading model")
    model_name = "dangvantuan/sentence-camembert-base"
    encoder = SentenceTransformer(model_name, device="cuda")
    encoder.eval()

    print("Loading data from ./data/processed/15th_merged_data_short.pkl")
    df = pd.read_pickle("./data/processed/15th_merged_data_short.pkl")
    df = df[
        [
            "nom",
            "groupe",
            "date_seance",
            "nb_mots_approx",
            "profession",
            "titre",
            "titre_complet",
            "intervention",
            "sexe",
            "n_y_naissance",
            "label",
        ]
    ]
    reg = "(article|l'article)\s*(\d+[^\w\s]*|premier|deuxième|troisième|[^\w\s]*\d+[^\w\s]*)"
    df["titre_regexed"] = df["titre"].str.replace(reg, "Article X", regex=True)
    df["contexte"] = (
        df["titre_complet"]
        .str.split(" > ")
        .apply(lambda x: x[0] if len(x) > 1 else "Sans contexte")
    )

    deputes = ["Jean-Luc Mélenchon", "Marine Le Pen"]
    groupes = ["LFI", "LR", "NI"]

    idx = df[(df.groupe.isin(groupes)) & (df.nb_mots_approx < 64) & (df.nb_mots_approx > 16)].index

    print("Creating the embeddings")
    docs = df.loc[idx, "intervention"].tolist()
    labels = df.loc[idx, "label"].tolist()
    sexes = df.loc[idx, "sexe"].map({"H": 0.0, "F": 1.0}).tolist()
    naissance = df.loc[idx, "n_y_naissance"].tolist()
    titre = encoder.encode(
        df.loc[idx, "titre_regexed"].tolist(), batch_size=128, show_progress_bar=True, device="cuda"
    )
    contexte = encoder.encode(
        df.loc[idx, "contexte"].tolist(), batch_size=128, show_progress_bar=True, device="cuda"
    )
    embs = encoder.encode(
        df.loc[idx, "intervention"].tolist(), batch_size=128, show_progress_bar=True, device="cuda"
    )

    print("Fitting the topic model")
    topic_model = BERTopic(
        embedding_model=encoder, top_n_words=10, calculate_probabilities=True, verbose=True
    )
    topics, probs = topic_model.fit_transform(docs, embeddings=embs)

    print("Showing some results")
    freq = topic_model.get_topic_info()
    print(freq.head(25))

    print(f"Saving the outputs in {RESULT_DIR}")
    print("Saving the scatter plot in topic_modelling_embeddings_ex.html")
    emb_fig = topic_model.visualize_documents(docs, embeddings=embs)
    emb_fig.write_html(RESULT_DIR / "topic_modelling_embeddings_ex.html", include_plotlyjs="cdn")
    emb_fig.write_image(RESULT_DIR / "topic_modelling_embeddings_ex.png")

    print("Saving the main topics topic_modelling_words_ex.html")
    words_fig = topic_model.visualize_barchart(top_n_topics=16, n_words=10)
    words_fig.write_html(RESULT_DIR / "topic_modelling_words_ex.html", include_plotlyjs="cdn")
    words_fig.write_image(RESULT_DIR / "topic_modelling_words_ex.png")

    topic_df = topic_model.get_document_info(docs)
    print(topic_df[topic_df["Topic"] != -1].head(20))
