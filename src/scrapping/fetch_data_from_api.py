# -*- coding: utf-8 -*-

# Copyright (c) 2022 Gabriel WATKINSON and Jéremie STYM-POPPER
# SPDX-License-Identifier: MIT

"""This module contains functions for working with data about deputies and
their interventions.

Examples:
    To have the default scrapping use :

        $ python src/scrapping/fetch_data_from_api.py

    Else, you can specify some parameters :

        $ python src/scrapping/fetch_data_from_api.py --legislature="2017-2022" --verbose=1
"""

import pickle
from pathlib import Path

import click
import pandas as pd
from tqdm.autonotebook import tqdm

from src.scrapping.api import CPCApi


# Intermediary functions
def deputies_of_group(deputies_df, group, n_deputies=None):
    """Returns the list of deputies belonging to the specified group.

    Args:
        deputies_df (pandas.DataFrame): DataFrame containing information about deputies.
        group (str): Group name to filter by.
        n_deputies (int, optional): Number of deputies to return. If not provided,
        all deputies in the group will be returned.

    Returns:
        List[str]: List of deputy names belonging to the specified group.
    """
    all_names = deputies_df[deputies_df["groupe_sigle"] == group]["nom"]
    res = all_names[:n_deputies] if n_deputies else all_names
    return res


def title_spliter(title_interventions):
    """Split intervention titles and dates from a list of strings.

    Args:
        title_interventions (List[str]): List of strings containing titles and dates, separated
        by either " : " or " - ".

    Returns:
        Tuple[List[str], List[str]]: A tuple containing two lists: the first contains the titles,
        the second contains the dates.

    Raises:
        ValueError: If the title or date format is not compliant.
    """
    titles = []
    dates = []
    for text in title_interventions:
        if ":" in text:
            date, title = text.split(" : ")
        elif "-" in text:
            title, date = text.split(" - ")
        else:
            raise ValueError("Title not compliant")
        titles.append(title)

        if " le " in date:
            dates.append(date.split(" le ")[1])
        elif " du " in date:
            dates.append(date.split(" du ")[1])
        else:
            raise ValueError("Date format not compliant")

    return titles, dates


def interventions_of_group(deputies_df, apidep, group, verbose=False, n_deputies=None):
    """Returns a list of 50 interventions by deputies belonging to the
    specified group.

    Args:
        deputies_df (pandas.DataFrame): DataFrame containing information about deputies.
        apidep: API object for retrieving interventions data.
        group (str): Acronym of the group to filter by.
        n_deputies (int, optional): Maximum number of selected deputies in the group.
            If not provided, all deputies in the group will be returned.
        verbose (bool, optional): Whether to display a progress bar or not.

    Returns:
        pandas.DataFrame: A DataFrame containing the group acronym, the deputy's name,
            the session titles, the session dates, and the interventions.
    """
    # Get the names of the deputies belonging to the specified group
    names = deputies_of_group(deputies_df, group, n_deputies)

    # Initialize an empty list to store the interventions data
    interventions = []

    # Iterate over each deputy in the group
    pbar = tqdm(names, leave=False) if verbose else names
    for name in pbar:
        # Get the interventions and session titles for the current deputy
        list_interventions = apidep.interventions(name)
        title_interventions = apidep.session_title(name)

        # Split the session titles and dates from the title_interventions list
        session_title, session_date = title_spliter(title_interventions)

        # Append the group acronym, deputy's name, session titles, session dates,
        # and interventions to the interventions list
        interventions.append([group, name, session_title, session_date, list_interventions])

    # Convert the interventions data to a DataFrame
    interventions_df = pd.DataFrame(
        interventions,
        columns=["groupe", "nom", "session_title", "session_date", "interventions"],
    )

    return interventions_df


def main(
    legislature=None,
    groups=None,
    path="./dataframes/",
    filename="df.pickle",
    verbose=True,
):
    """This script creates a huge dataframe with all deputies, all groups, and
    50 interventions for each.

    Warning: this script may take a long time to run, do not run it unless necessary.
    """
    # Initialize the API object
    apidep = CPCApi(legislature=legislature)

    # Retrieve deputies data from the API
    deputies_df = apidep.deputies_df
    groupes = apidep.groups if groups is None else groups

    # Initialize an empty DataFrame to store the interventions data
    interventions_df = pd.DataFrame(
        columns=["groupe", "nom", "session_title", "session_date", "interventions"]
    )

    # Iterate over each group
    try:
        pbar = tqdm(groupes) if tqdm else groupes
        for groupe in pbar:
            interventions_df = pd.concat(
                [
                    interventions_df,
                    interventions_of_group(deputies_df, apidep, groupe, verbose=verbose),
                ],
                ignore_index=True,
            )
    except Exception as e:
        print(e)
        if len(interventions_df) == 0:
            return
        else:
            click.echo("Saving current incomplete dataframe ...")
            filename = "incomplete" + filename

    # Create the specified directory if it doesn't exist
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    prefix = f"{legislature.replace('-', '_')}_" if legislature else "last_legislature_"
    filename = prefix + filename
    exploded_filename = "exploded_" + filename

    # Save the interventions_df DataFrame to a pickle file
    click.echo(f"Saving dataframe in {filename} ...")
    with open(path / filename, "wb") as f:
        pickle.dump(interventions_df, f)

    # Tidy the data by exploding the lists in the interventions_df DataFrame
    interventions_df_tidy = interventions_df.explode(
        ["session_title", "session_date", "interventions"], ignore_index=True
    ).astype(str)

    # Save the interventions_df_tidy DataFrame to a pickle file
    click.echo(f"Saving exploded dataframe in {filename} ...")
    with open(path / exploded_filename, "wb") as f:
        pickle.dump(interventions_df_tidy, f)


@click.command("scrap_api", context_settings={"show_default": True})
@click.option(
    "--legislature",
    default=None,
    help="Legislature to filter by (in 2007-2012, 2012-2017, 2017-2022 or None).",
)
@click.option(
    "--groups",
    default=None,
    help="List of groups to scrap. If not provided, all groups will be scrapped.",
)
@click.option(
    "--path",
    default="./dataframes/",
    help="Path to the directory where the pickle file will be saved.",
)
@click.option("--filename", default="df.pickle", help="Name of the pickle file.")
@click.option("--verbose", default=True, help="Whether to display a progress bar or not.")
def main_click(legislature, groups, path, filename, verbose):
    """This script creates a huge dataframe with all deputies, all groups, and
    50 interventions for each.

    Warning: this script may take a long time to run, do not run it unless necessary.
    """
    main(legislature, groups, path, filename, verbose)


if __name__ == "__main__":
    main_click()
