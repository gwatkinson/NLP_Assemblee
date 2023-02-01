"""Module to process the raw data.

It uses compiled data from the Assemblée Nationale website, and
transform it into a dataframe used in the modelisation.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.autonotebook import tqdm
from transformers import BertTokenizer, CamembertTokenizer


class DataProcessing:
    def __init__(
        self,
        deputies_df_path: str,
        compiled_data_path: str,
        label_dict: dict = {},
        process: bool = True,
        tokenize: bool = True,
        save: str | bool = "./processed_data",
        load: str | bool = False,
        max_len: int = 512,
        year_min: int = 1940,
        year_max: int = 2022,
        legislature: int = 15,
        verbose: bool = True,
    ) -> None:
        """Initialize the data processing.

        Args:
            deputies_df_path (str): Path to the raw dataframe with deputies information
            compiled_data_path (str): Path to the raw dataframe with deputies interventions
            process (bool, optional): If True, process the data.
            save (str | bool, optional): If str, save the data in the given path.
                If False, don't save the processed data.
            legislature (int, optional): Number of the legislature to process,
                for the file name during saving.
        """
        self.deputies_df_path = deputies_df_path
        self.compiled_data_path = compiled_data_path
        self.process = process
        self.save = save
        self.load = load
        self.tokenize = tokenize
        self.max_len = max_len
        self.year_min = year_min
        self.year_max = year_max
        self.legislature = legislature
        self.label_dict = label_dict
        self.verbose = verbose

        self.deputies_df = pd.read_pickle(self.deputies_df_path)
        self.compiled_data = pd.read_csv(self.compiled_data_path, sep="\t")

        if load:
            print("Loading the processed data...")
            try:
                self.load_tables()
            except Exception as e:
                print("Error while loading the processed data.")
                print(e)
                self.load = False

        if process and not self.load:
            print("Cleaning the deputies dataframe...")
            self.clean_deputies_df()  # Create self.deputies_df_processed
            print("Cleaning the compiled dataset...")
            self.clean_compiled_data()  # Create self.compiled_data_processed
            print("Merging the two dataframes...")
            self.merging_process()  # Create self.processed_data

        if tokenize and not self.load:
            print("Tokenizing the resulting data...")
            self.tokenizing_process()  # Create self.records

        if save and not self.load:
            print("Saving the processed data...")
            self.save_tables()

    @staticmethod
    def process_intervention(row, max_len=512):
        """Remove the html tags from intervention texts.

        Args:
            row (pandas.Series): A deputy/session/intervention row
            max_len (int, optional): Max number of words before split in paragraph. Defaults to 256.

        Returns:
            str: Clean text of the intervetion from one deputy
        """
        if row["nb_mots"] > max_len:
            paragraphs = BeautifulSoup(row["intervention"], features="lxml").find_all("p")
            return [_.getText() for _ in paragraphs]
        return BeautifulSoup(row["intervention"], features="lxml").getText()

    def clean_compiled_data(self):
        """Clean the compiled data by removing some noising rows.

        Apply the process_intervention

        Returns:
            compiled_data_processed (pandas.DataFrame): Compiled dataframe with clean interventions

        Creates:
            self.compiled_data_processed (pandas.DataFrame): Compiled dataframe with clean
                interventions
        """
        tdf = self.compiled_data.copy()

        tdf = tdf.rename(
            columns={
                "nom": "personnalite",
                "parlementaire": "depute",
                "sexe": "depute_sexe",
                "parlementaire_groupe_acronyme": "depute_groupe",
                "parlementaire_groupe": "depute_groupe",
                "section": "titre_complet",
                "sous_section": "titre",
            }
        )

        # Remove the president
        tdf_without_pres = tdf[
            (~tdf.fonction.isin(["président", "présidente", "président, doyen d'âge"]))
        ]

        tdf_without_exclamations = tdf_without_pres[
            # Remove the interventions without title (transitions, presidents, ...)
            (~tdf_without_pres.titre.isna())
            &
            # Remove the shortest intervention (not useful for classification)
            (tdf_without_pres.nb_mots > 7)
        ]

        intervention_count = (
            tdf_without_exclamations["depute"]
            .value_counts()
            .rename("intervention_count")
            .to_frame()
        )
        tdf_without_exclamations = tdf_without_exclamations.merge(
            intervention_count, left_on="depute", right_index=True
        )

        processed_tdf_intervention = tdf_without_exclamations.apply(
            lambda row: self.process_intervention(row, max_len=self.max_len), axis=1
        )
        tdf_intervention_processed = tdf_without_exclamations.assign(
            intervention=processed_tdf_intervention
        ).explode("intervention")
        tdf_intervention_processed = tdf_intervention_processed[
            tdf_intervention_processed.intervention.str.count(" ") > 3
        ]
        tdf_intervention_processed["nb_mots_approx"] = (
            tdf_intervention_processed.intervention.str.count(" ") + 1
        )

        tdf_without_short = tdf_intervention_processed[
            tdf_intervention_processed["intervention_count"] > 10
        ].reset_index(drop=True)

        tdf_without_short["intervention"] = tdf_without_short["intervention"].str.replace(
            "\x92", "'"
        )

        self.compiled_data_processed = tdf_without_short

        return tdf_without_short

    def clean_deputies_df(self):
        """Clean the dataframe with deputies information.

        Returns:
            deputies_df_processed (pandas.DataFrame): Return clean dataframe on deputy information

        Creates:
            self.deputies_df_processed (pandas.DataFrame): Clean deputies dataframe
        """
        clean_deputies_df = self.deputies_df.copy()
        clean_deputies_df["profession"].fillna("None", inplace=True)

        self.deputies_df_processed = clean_deputies_df

        return clean_deputies_df

    def add_features(self, processed_df):
        processed_df["date"] = pd.to_datetime(processed_df.date_seance)
        processed_df["year"] = processed_df["date"].dt.year
        processed_df["month"] = processed_df["date"].dt.month
        processed_df["day"] = processed_df["date"].dt.day
        processed_df["y_naissance"] = pd.to_datetime(processed_df.date_naissance).dt.year
        year_norm_const = self.year_max - self.year_min
        processed_df["n_y_naissance"] = (
            self.year_max - processed_df["y_naissance"]
        ) / year_norm_const

        leg_start = processed_df["year"].min()
        processed_df["n_year"] = (processed_df["year"] - leg_start) / 5
        processed_df["cos_month"] = np.cos(2 * np.pi * processed_df["month"] / 12)
        processed_df["sin_month"] = np.sin(2 * np.pi * processed_df["month"] / 12)
        processed_df["cos_day"] = np.cos(2 * np.pi * processed_df["day"] / 31)
        processed_df["sin_day"] = np.sin(2 * np.pi * processed_df["day"] / 31)

        processed_df["n_sexe"] = processed_df["sexe"].map({"H": 0, "F": 1})
        processed_df["label"] = processed_df["groupe"].map(self.label_dict)

        return processed_df

    def merging_process(self):
        """Create the final dataframe with deputy interventions and global
        information.

        Merge the information of the deputy interventions with global
        information on the deputies Group the interventions by seance information
        and deputy characteristics.

        Returns:
            processed_data (pandas.DataFrame): Final processed dataframe

        Creates:
            self.processed_data (pandas.DataFrame): Final processed dataframe
            self.short_interventions (pandas.DataFrame): The unmerged final dataframe
        """
        tdf_processed = self.compiled_data_processed.copy()
        clean_deputies = self.deputies_df_processed[
            ["nom", "date_naissance", "sexe", "profession", "nb_mandats"]
        ].copy()

        simple_interventions = tdf_processed[
            [
                "depute",
                "depute_groupe",
                "seance_id",
                "date",
                "titre",
                "titre_complet",
                "intervention",
                "nb_mots",
                "intervention_count",
                "nb_mots_approx",
            ]
        ].rename(
            columns={
                "depute": "nom",
                "depute_groupe": "groupe",
                "date": "date_seance",
            }
        )

        grouped_df = (
            simple_interventions.groupby(
                [
                    "seance_id",
                    "date_seance",
                    "nom",
                    "groupe",
                    "titre",
                    "titre_complet",
                    "intervention_count",
                ]
            )
            .agg(
                intervention=pd.NamedAgg(
                    column="intervention", aggfunc=lambda group: " ".join(group)
                ),
                nb_mots=pd.NamedAgg(column="nb_mots", aggfunc="sum"),
                nb_mots_approx=pd.NamedAgg(column="nb_mots_approx", aggfunc="sum"),
            )
            .reset_index()
        )

        processed_df = grouped_df.merge(clean_deputies, left_on="nom", right_on="nom")
        processed_df = self.add_features(processed_df)

        short_interventions = simple_interventions.merge(
            clean_deputies, left_on="nom", right_on="nom"
        )
        short_interventions = self.add_features(short_interventions)

        self.processed_data = processed_df
        self.short_interventions = short_interventions

        return processed_df

    def tokenizing_process(self):
        """Tokenize the interventions with Camembert and BERT tokenizers.

        Returns:
            list[dict[str, Any]]: The list of records with the tokenized interventions and all
                the other information.

        Creates:
            self.records (list[dict[str, Any]]): The list of records with the tokenized
                interventions and all the other information.
        """
        self.records = self.short_interventions.to_dict(orient="records")
        self.camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
        self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

        pbar = tqdm(self.records) if self.verbose else self.records
        for record in pbar:
            record["camembert_tokens"] = {
                "intervention": self.camembert_tokenizer.encode(
                    record["intervention"],
                    add_special_tokens=True,
                    truncation=True,
                ),
                "titre_complet": self.camembert_tokenizer.encode(
                    record["titre_complet"],
                    add_special_tokens=True,
                    truncation=True,
                ),
                "titre": self.camembert_tokenizer.encode(
                    record["titre"],
                    add_special_tokens=True,
                    truncation=True,
                ),
                "profession": self.camembert_tokenizer.encode(
                    record["profession"],
                    add_special_tokens=True,
                    truncation=True,
                ),
            }
            record["bert_tokens"] = {
                "intervention": self.bert_tokenizer.encode(
                    record["intervention"],
                    truncation=True,
                ),
                "titre_complet": self.bert_tokenizer.encode(
                    record["titre_complet"],
                    truncation=True,
                ),
                "titre": self.bert_tokenizer.encode(
                    record["titre"],
                    truncation=True,
                ),
                "profession": self.bert_tokenizer.encode(
                    record["profession"],
                    truncation=True,
                ),
            }

        return self.records

    def save_tables(self):
        """Save the generated tables."""
        path = Path(self.save)
        path.mkdir(parents=True, exist_ok=True)
        self.compiled_data_processed.to_pickle(
            path / f"{self.legislature}th_compiled_processed.pkl"
        )
        self.deputies_df_processed.to_pickle(path / f"{self.legislature}th_deputies_processed.pkl")
        self.processed_data.to_pickle(path / f"{self.legislature}th_merged_data.pkl")
        self.short_interventions.to_pickle(path / f"{self.legislature}th_merged_data_short.pkl")
        try:
            with open(path / f"{self.legislature}th_records.pkl", "wb") as f:
                pickle.dump(self.records, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print("Could not save records.")
            print(e)
        with open(path / f"{self.legislature}th_bert_tokenizer.pkl", "wb") as f:
            pickle.dump(self.bert_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path / f"{self.legislature}th_camembert_tokenizer.pkl", "wb") as f:
            pickle.dump(self.camembert_tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_pickle_df(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        else:
            return pd.read_pickle(path)

    @staticmethod
    def load_pickle_file(path):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")
        else:
            with open(path, "rb") as f:
                tmp = pickle.load(f)
            return tmp

    def load_tables(self):
        """Load the generated tables."""
        path = Path(self.load)

        if not path.exists():
            raise FileNotFoundError(f"Path {path} does not exist.")

        print("Loading the dataframes...")
        self.compiled_data_processed = self.load_pickle_df(
            path / f"{self.legislature}th_compiled_processed.pkl"
        )
        self.deputies_df_processed = self.load_pickle_df(
            path / f"{self.legislature}th_deputies_processed.pkl"
        )
        self.processed_data = self.load_pickle_df(
            path / f"{self.legislature}th_merged_data_short.pkl"
        )
        self.short_interventions = self.load_pickle_df(
            path / f"{self.legislature}th_merged_data_short.pkl"
        )

        print("Loading the records...")
        self.records = self.load_pickle_file(path / f"{self.legislature}th_records.pkl")

        print("Loading the tokenizers...")
        self.bert_tokenizer = self.load_pickle_file(
            path / f"{self.legislature}th_bert_tokenizer.pkl"
        )
        self.camembert_tokenizer = self.load_pickle_file(
            path / f"{self.legislature}th_camembert_tokenizer.pkl"
        )

    def load_tokenizers(self):
        try:
            path = Path(self.load)
            self.bert_tokenizer = self.load_pickle_file(
                path / f"{self.legislature}th_bert_tokenizer.pkl"
            )
            self.camembert_tokenizer = self.load_pickle_file(
                path / f"{self.legislature}th_camembert_tokenizer.pkl"
            )
        except FileNotFoundError:
            self.camembert_tokenizer = CamembertTokenizer.from_pretrained("camembert-base")
            self.bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
