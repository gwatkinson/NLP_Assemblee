"""Module to process the raw data.

It uses compiled data from the Assemblée Nationale website, and
transform it into a dataframe used in the modelisation.
"""

import pandas as pd
from bs4 import BeautifulSoup


def process_intervention(row, max_len=256):
    """Remove the html tags from intervention texts.

    Args:
        row (pandas.Series): A deputy/session/intervention row
        max_len (int, optional): Max number of words before split in paragraph. Defaults to 256.

    Returns:
        str: Clean text of the intervetion from one deputy
    """
    if row["nb_mots"] > max_len:
        paragraphs = BeautifulSoup(row["intervention"]).find_all("p")
        return [_.getText() for _ in paragraphs]
    return BeautifulSoup(row["intervention"]).text


def table_cleaner(tdf):
    """Clean the data by removing some noising rows.

    Apply the process_intervention

    Args:
        tdf (pandas.DataFrame): The dataframe to clean

    Returns:
        pandas.DataFrame: Clean dataframe with clean interventions
    """
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
        tdf_without_exclamations["depute"].value_counts().rename("intervention_count").to_frame()
    )
    tdf_without_exclamations = tdf_without_exclamations.merge(
        intervention_count, left_on="depute", right_index=True
    )

    processed_tdf_intervention = tdf_without_exclamations.apply(process_intervention, axis=1)
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

    return tdf_without_short


def deputy_info(deputies_df):
    """Return a clean data for information on deputies Fill with NoneValue for
    NaN profession.

    Args:
        deputies_df (pandas.DataFrame): Raw dataframe with deputy information

    Returns:
        pandas.DataFrame: Return clean dataframe on deputy information
    """
    clean_deputies_df = deputies_df[
        ["nom", "date_naissance", "num_circo", "profession", "nb_mandats"]
    ].copy()
    clean_deputies_df["profession"].fillna("None", inplace=True)

    return clean_deputies_df


def merging_process(tdf, deputies_df):
    """Merge the information of the deputy interventions with global
    information on the deputies Group the interventions by seance information
    and deputy characteristics.

    Args:
        tdf (pandas.DataFrame): Raw dataframe (unprocessed) gathering deputy interventions
        deputies_df (pandas.DataFrame): Raw dataframe (unprocessed) gathering global
            information on deputies

    Returns:
        pandas.DataFrame: Clean dataframe with interventions grouped by seance and deputy
            information
    """
    tdf_processed = table_cleaner(tdf)

    simple_interventions = tdf_processed[
        ["depute", "depute_groupe", "seance_id", "date", "titre", "titre_complet", "intervention"]
    ].rename(
        columns={
            "depute": "nom",
            "depute_groupe": "groupe",
            "date": "date_seance",
        }
    )

    grouped_df = (
        simple_interventions.groupby(
            ["seance_id", "date_seance", "nom", "groupe", "titre", "titre_complet"]
        )
        .agg(intervention=pd.NamedAgg(column="intervention", aggfunc=lambda group: " ".join(group)))
        .reset_index()
    )

    clean_deputies = deputy_info(deputies_df)

    processed_df = grouped_df.merge(clean_deputies, left_on="nom", right_on="nom")

    return processed_df
