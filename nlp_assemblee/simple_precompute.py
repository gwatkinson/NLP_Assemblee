import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import torch
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from tqdm.autonotebook import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask, pooled_output="mean"):
    if pooled_output == "mean":
        token_embeddings = model_output[
            "last_hidden_state"
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        outs = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )
    elif pooled_output == "cls":
        outs = token_embeddings = model_output["last_hidden_state"][:, 0]
    elif pooled_output == "pooled":
        outs = model_output["pooler_output"]
    return outs


def get_embeddings_list(model, df, var, batch_size=64):
    embedding_list = []
    values = df[var]
    for i in tqdm(range(0, len(values), batch_size)):
        batch = values[i : i + batch_size].to_list()
        encoded_var = model.encode(batch)
        embedding_list.extend([encoded_var[v, :] for v in range(len(batch))])

    return np.array(embedding_list)


def get_embeddings_dict(model, df, var, batch_size=64):
    embedding_dict = {}
    unique_values = df[var].unique()
    label_to_int = {k: v for v, k in enumerate(unique_values)}
    for i in tqdm(range(0, len(unique_values), batch_size)):
        batch = unique_values[i : i + batch_size]
        encoded_var = model.encode(batch)
        embedding_dict.update({label_to_int[k]: encoded_var[v, :] for v, k in enumerate(batch)})

    return embedding_dict, label_to_int


def get_embeddings_matrix(model, df, var, batch_size=64):
    unique_values = list(df[var].unique())
    label_to_int = {k: v for v, k in enumerate(unique_values)}
    encoded_var = model.encode(unique_values, batch_size=batch_size, show_progress_bar=True)

    return encoded_var, label_to_int


def get_embeddings_list_unbatched(model, df, var, batch_size=64):
    values = list(df[var])
    embedding_list = model.encode(values, batch_size=batch_size, show_progress_bar=True)

    return embedding_list


def get_embeddings_list_from_hugging(
    model, tokenizer, df, var, batch_size=64, pooled_output="mean"
):
    embedding_list = []
    values = df[var]
    for i in tqdm(range(0, len(values), batch_size)):
        batch = values[i : i + batch_size].to_list()
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        with torch.no_grad():
            model_output = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(pooled_output == "cls"),
            )
        sentence_embeddings = (
            mean_pooling(model_output, attention_mask, pooled_output=pooled_output).cpu().numpy()
        )
        embedding_list.extend([sentence_embeddings[v, :] for v in range(len(batch))])

    del input_ids, attention_mask, model_output
    torch.cuda.empty_cache()
    return np.array(embedding_list)


def get_embeddings_dict_from_hugging(
    model, tokenizer, df, var, batch_size=64, pooled_output="mean"
):
    embedding_dict = {}
    unique_values = df[var].unique()
    label_to_int = {k: v for v, k in enumerate(unique_values)}
    for i in tqdm(range(0, len(unique_values), batch_size)):
        batch = unique_values[i : i + batch_size].tolist()
        encoded_input = tokenizer(
            batch, padding=True, truncation=True, return_tensors="pt", max_length=512
        )
        input_ids = encoded_input["input_ids"].to(device)
        attention_mask = encoded_input["attention_mask"].to(device)
        with torch.no_grad():
            model_output = model(
                input_ids,
                attention_mask=attention_mask,
                output_hidden_states=(pooled_output == "cls"),
            )
        sentence_embeddings = (
            mean_pooling(model_output, attention_mask, pooled_output=pooled_output).cpu().numpy()
        )
        embedding_dict.update(
            {label_to_int[k]: sentence_embeddings[v, :] for v, k in enumerate(batch)}
        )

    del input_ids, attention_mask, model_output
    torch.cuda.empty_cache()
    return embedding_dict, label_to_int


def plot_proj_from_emb_dict(
    projection, embedding_dict, label_to_int, df, var, path, width=1200, height=1200
):
    embedding_df = pd.DataFrame(embedding_dict).T
    int_to_label = {v: k for k, v in label_to_int.items()}

    if projection == "umap":
        fit = umap.UMAP()
    elif projection == "tsne":
        fit = TSNE(n_components=2, random_state=0)
    elif projection == "pca":
        fit = PCA(n_components=2)

    proj = fit.fit_transform(embedding_df)
    proj_df = pd.DataFrame(proj, columns=[f"{projection}-x", f"{projection}-y"])
    proj_df[var] = embedding_df.index
    proj_df[var] = proj_df[var].map(int_to_label)
    proj_df["Orientation"] = df.groupby(var)["label"].mean().loc[proj_df[var]].values

    fig = px.scatter(
        proj_df,
        x=f"{projection}-x",
        y=f"{projection}-y",
        color="Orientation",
        hover_name=var,
        width=width,
        height=height,
        color_continuous_scale=px.colors.diverging.Temps,
    )

    path = Path(path) / var / "images"
    path.mkdir(exist_ok=True, parents=True)
    fig.write_image(path / f"{projection}.png")
    fig.write_html(path / f"{projection}.html")

    return fig


def save_embedding_matrix(embedding_dict, label_to_int, var, path):
    int_to_label = {v: k for k, v in label_to_int.items()}

    embs = []
    for k, v in embedding_dict.items():
        embs.append(v)
    embs = np.vstack(embs)

    res = {
        "embeddings": embs,
        "int_to_label": int_to_label,
        "label_to_int": label_to_int,
    }

    path = Path(path) / var
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "embeddings.pkl", "wb") as f:
        pickle.dump(res, f)


def save_embedding_matrix_from_list(embedding_matrix, label_to_int, var, path):
    int_to_label = {v: k for k, v in label_to_int.items()}
    res = {
        "embeddings": embedding_matrix,
        "int_to_label": int_to_label,
        "label_to_int": label_to_int,
    }

    path = Path(path) / var
    path.mkdir(exist_ok=True, parents=True)
    with open(path / "embeddings.pkl", "wb") as f:
        pickle.dump(res, f)


def train_test_val_split(labels, train_pc=0.5, val_pc=0.2, stratify=True, random_state=42):
    idx = np.arange(len(labels))

    idx_train, idx_test = train_test_split(
        idx, train_size=train_pc, stratify=labels if stratify else None, random_state=random_state
    )
    idx_val, idx_test = train_test_split(
        idx_test,
        train_size=val_pc / (1 - train_pc),
        stratify=labels[idx_test] if stratify else None,
        random_state=random_state,
    )

    return idx_train, idx_val, idx_test


# def precompute_embeddings(config_file, bert_type,
# data_folder="./", output_folder="./data/precomputed", device=device):
#     datasets, loaders = build_dataset_and_dataloader_from_config(config_file, data_folder)

#     if bert_type == "bert":
#         embedder = bert_type

#     for phase in tqdm(["train", "val", "test"]):
#         embeddings = {
#             "intervention": [],
#             "titre_complet": [],
#             "profession": [],
#             "features": [],
#             "label": [],
#         }
#         for x, y in tqdm(loaders[phase]):
#             x = {k: v.to(device) for k, v in x.items()}
#             with torch.no_grad():
#                 for k, v in x.items():
#                     if k != "features":
#                         embeddings[k].append(embedder[k].bert(v)["pooler_output"].cpu().numpy())
#                     else:
#                         embeddings["features"].append(v.cpu().numpy())
#             embeddings["label"].append(y.cpu().numpy())

#         embs = {
#             "intervention": np.vstack(embeddings["intervention"]),
#             "titre_complet": np.vstack(embeddings["titre_complet"]),
#             "profession": np.vstack(embeddings["profession"]),
#             "features": np.vstack(embeddings["features"]),
#             "label": np.hstack(embeddings["label"]),
#         }

#         output_folder = Path(output_folder)
#         output_folder.mkdir(exist_ok=True, parents=True)
#         output_file = output_folder / f"{bert_type}_embeddings_{phase}.pkl"
#         with open(output_file, "wb") as f:
#             pickle.dump(embs, f)
