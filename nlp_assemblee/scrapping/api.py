# -*- coding: utf-8 -*-

"""A Python wrapper for the Regards Citoyens nosdeputes.fr and nossenateurs.fr
APIs.

These APIs provide information about French parliamentary deputies, such
as their names, parties, and parliamentary interventions.
"""

import requests

# from grequests import async
import aiohttp
import asyncio
import warnings


import json
import re
import bs4
import pandas as pd
from urllib import request
import unidecode
from tqdm.autonotebook import tqdm
from pathlib import Path

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from fuzzywuzzy.process import extractBests

__all__ = ["CPCApi"]


def memoize(f):
    cache = {}

    def aux(*args, **kargs):
        k = (args, tuple(sorted(kargs.items())))
        if k not in cache:
            cache[k] = f(*args, **kargs)
        return cache[k]

    return aux


class CPCApi(object):
    """Class for interacting with the CPC API."""

    def __init__(self, ptype="depute", legislature=None, format="json"):
        """Initializes the API object.

        Args:
            ptype (str, optional): The type of parliamentarian to retrieve data for.
                Valid values are "depute" (deputy) or "senateur" (senator).
                Default: "depute"
            legislature (str, optional): The legislature to retrieve data for.
                Valid values are "2007-2012" or "2012-2017", "2017-2022",
                or None for current legislature.
                Default: None
            format (str, optional): The format to retrieve data in.
                Default: "json"

        Raises:
            AssertionError: If ptype is not "depute" or "senateur", or if legislature is not
                "2007-2012", "2012-2017", "2017-2022", or None.
        """
        assert ptype in ["depute", "senateur"]
        assert legislature in ["2007-2012", "2012-2017", "2017-2022", None]
        self.legislature = legislature
        self.legislature_name = "last" if legislature is None else legislature
        self.format = format
        self.ptype = ptype
        self.ptype_plural = ptype + "s"
        self.prefix = "www" if legislature is None else legislature
        self.base_url = f"https://{self.prefix}.nos{self.ptype_plural}.fr"

        # Fetches the list of deputies
        self.parlementaires()  # creates self.parlementaires_list

        # Create the dataframe of deputies and the name of the groups
        self.get_deputies_df()  # creates self.deputies_df and self.groups and self.deputies

    def synthese(self, month=None):
        """Retrieves a summary of parliamentary activity.

        Args:
            month: The month for which to retrieve the summary (default: None).
                The month should be in the format "YYYYMM".

        Returns:
            A list of dictionaries, each containing information about a parliamentarian.

        Raises:
            AssertionError: If `month` is None and the legislature is "2012-2017".
        """
        # Check if the month is specified and if the legislature is "2012-2017"
        if month is None and self.legislature == "2012-2017":
            # Raise an error if both conditions are true
            raise AssertionError(
                "Global Synthesis on legislature does not work,"
                + "see https://github.com/regardscitoyens/nosdeputes.fr/issues/69"
            )

        # Set the month to "data" if it is not specified
        if month is None:
            month = "data"

        # Retrieve the summary from the API
        url = f"{self.base_url}/synthese/{month}/{self.format}"
        data = requests.get(url).json()

        # Extract the parliamentarians from the response and return them
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def parlementaire(self, slug_name):
        """Retrieves information about a parliamentarian.

        Args:
            slug_name: The slug name of the parliamentarian.

        Returns:
            A dictionary of information about the parliamentarian.
        """
        # Build the URL for the parliamentarian's information
        url = f"{self.base_url}/{slug_name}/{self.format}"

        # Retrieve the information from the API and return it
        return requests.get(url).json()[self.ptype]

    def picture_url(self, slug_name, pixels="60"):
        """Builds the URL for a picture of a parliamentarian.

        Args:
            slug_name: The slug name of the parliamentarian.
            pixels: The size of the picture to retrieve (default: 60).

        Returns:
            The URL for the picture.
        """
        # Build the URL for the picture
        return f"{self.base_url}/{self.ptype}/photo/{slug_name}/{pixels}"

    def picture(self, slug_name, pixels="60"):
        """Retrieves a picture of a parliamentarian.

        Args:
            slug_name: The slug name of the parliamentarian.
            pixels: The size of the picture to retrieve (default: 60).

        Returns:
            A request object for the picture.
        """
        # Build the URL for the picture
        url = self.picture_url(slug_name, pixels=pixels)

        # Retrieve the picture from the API and return it
        return requests.get(url)

    def search(self, q, page=1):
        """Searches for parliamentaries.

        Args:
            q: The query to search for.
            page: The page of results to retrieve (default: 1).

        Returns:
            The raw data of the search results in CSV format.
        """
        # Build the URL
        url = f"{self.base_url}/recherche/{q}?page={page}&format=csv"

        # Retrieve the data from the API and return it
        return requests.get(url).content

    @memoize
    def parlementaires(self, active=None):
        """Retrieves a list of parliamentaries.

        Returns:
            A list of parliamentaries.
        """
        # Build the URL based on the active parameter
        if active is None:
            url = f"{self.base_url}/{self.ptype_plural}/{self.format}"
        else:
            url = f"{self.base_url}/{self.ptype_plural}/enmandat/{self.format}"

        # Retrieve the data from the API
        data = requests.get(url).json()

        # Extract the parliamentaries from the response and return them
        self.parlementaires_list = [depute[self.ptype] for depute in data[self.ptype_plural]]

        return self.parlementaires_list

    def get_deputies_df(self):
        """Retrieves a DataFrame of deputies information from the API.

        Returns:
            pandas.DataFrame: A DataFrame containing information about deputies.
        """
        # Retrieve deputies data from the API
        deputies_json = self.parlementaires_list

        # Convert the JSON data to a DataFrame
        cols_to_drop = [
            "sites_web",
            "emails",
            "adresses",
            "collaborateurs",
            "anciens_autres_mandats",
            "autres_mandats",
            "url_an",
            "id_an",
            "url_nosdeputes",
            "url_nosdeputes_api",
            "twitter",
        ]
        deputies_df = pd.json_normalize(deputies_json)
        deputies_df = deputies_df.drop(columns=cols_to_drop, errors="ignore")
        deputies_df["legislature"] = self.legislature_name

        self.deputies_df = deputies_df
        self.all_groups = deputies_df["groupe_sigle"].unique()
        self.deputies = deputies_df["nom"].unique()

    def search_parlementaires(self, q, field="nom", limit=5):
        """Searches for parliamentarians.

        Args:
            q: The query to search for.
            field: The field of the parliamentarian data to search in (default: "nom").
            limit: The maximum number of results to return (default: 5).

        Returns:
            A list of dictionaries, each containing information about a parliamentarian.
        """
        # Search for parliamentarians and return the best matches
        return extractBests(
            q,
            self.parlementaires_list,
            processor=lambda x: x[field] if type(x) == dict else x,
            limit=limit,
        )

    def get_deputy_interventions_list(
        self,
        dep_name,
        count=500,
        object_name="Intervention",
        sort=0,
        page=1,
        all_pages=False,
        max_interventions=1000,
    ):
        """Retrieves the urls of the interventions of a parliamentarian.

        See the documentation for the API for more information:
        https://github.com/regardscitoyens/nosdeputes.fr/blob/master/doc/api.md#r%C3%A9sultats-du-moteur-de-recherche

        Args:
            slug_name: The slug name of the parliamentarian.
                `prenom+nom` sans accents ni cédille et en remplaçant les espaces par des +,
                mais en conservant les traits d'union.
            count: The number of interventions to retrieve.
                Default: 500.
            object_name: The type of object to retrieve.
                Intervention, Amendement, QuestionEcrite, Commentaire, Parlementaire, Organisme.
                Default: "Intervention".
            sort: The sort order of the interventions, by pertinence or by date.
                Default: 0.
            page: The page of results to retrieve.
                Default: 1.
            all_pages: Whether to retrieve all pages of results.
                Default: False.

        Returns:
            A json object containing the links to the interventions.
        """
        if all_pages or max_interventions:
            page = 1

        # Remove accents and special characters from the name and replace spaces with +
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        slug_name = re.sub(" ", "+", unidecode.unidecode(name.lower()))

        # Build the URL for the search
        url = f"{self.base_url}/recherche?object_name={object_name}&format={self.format}"
        url += f"&tag=parlementaire%3D{slug_name}&count={count}&sort={sort}"
        page_param = f"&page={page}"

        # Retrieve the search results
        response = requests.get(url + page_param).json()
        response["dep_name"] = dep_name
        response["slug_name"] = slug_name
        response["legislature"] = self.legislature_name

        return response
        # Fetch other pages if necessary
        if all_pages:
            last_int = response["last_result"]
        else:
            last_int = min(max_interventions, response["last_result"])

        response["last_result"]
        while all_pages and response["end"] < last_int:
            page += 1
            page_param = f"&page={page}"

            tmp = requests.get(url + page_param).json()
            response["results"] += tmp["results"]
            response["end"] = tmp["end"]

        return response

    def get_all_interventions_urls(
        self,
        all_pages=True,
        max_interventions=1000,
        sort=0,
        count=500,
        object_name="Intervention",
        page=1,
        verbose=True,
        save="./data/interventions_urls_by_deputies.json",
    ):
        """Retrieves the URLs for all interventions by each deputy.

        Parameters:
            all_pages (bool): Whether to retrieve all pages of results or just the first page.
                Default is True.
            sort (int): An integer representing the sorting method to use.
                Default is 0.
            count (int): The number of results to retrieve per page.
                Default is 500.
            object_name (str): The name of the object to retrieve.
                Default is "Intervention".
            page (int): The page number to start at.
                Default is 1.
            verbose (bool): Whether to display a progress bar.
                Default is True.
            save (str): The file path to save the results to.
                Default is "./data/interventions_urls_by_deputies.json".

        Returns:
            deputies_list (dict): A dictionary where the keys are deputy names
                and the values are lists of URLs for their interventions.
        """
        deputies_list = {}

        # Get the interventions for each deputy
        pbar = tqdm(self.deputies, leave=False) if verbose else self.deputies
        for dep in pbar:
            res = self.get_deputy_interventions_list(
                dep_name=dep,
                count=count,
                object_name=object_name,
                sort=sort,
                page=page,
                all_pages=all_pages,
                max_interventions=max_interventions,
            )
            deputies_list[dep] = res

        # Save the results to a file if save path is provided
        if save:
            path = Path(save)
            path_list = list(path.parts)
            path_list.insert(-1, self.legislature_name)
            save_path = Path("").joinpath(*path_list[:-1])
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / path_list[-1], "w") as f:
                json.dump(deputies_list, f)
                self.deputies_list_file = str(save_path / path_list[-1])

        return deputies_list

    def fetch_interventions_of_deputy(
        self,
        dep_intervention_list,
        slug_name=None,
        max_interventions=1000,
        verbose=True,
        save="./data/",
    ):
        """Fetches the interventions from the API.

        Args:
            dep_intervention_list: A list of interventions.
                Returned from the `get_deputy_interventions_list` method.
            max_interventions: The maximum number of interventions to fetch.
                Default: 1000.
            verbose: Whether to display a progress bar.
                Default: True.

        Returns:
            A list of interventions.
        """
        # Initialize an empty list to store the interventions
        interventions = {
            "start": dep_intervention_list["start"],
            "end": dep_intervention_list["end"],
            "last_result": dep_intervention_list["last_result"],
            "interventions": [],
        }

        inters = (
            dep_intervention_list["results"][:max_interventions]
            if max_interventions
            else dep_intervention_list["results"]
        )
        # Fetch the interventions from the API
        pbar = tqdm(inters, leave=False) if verbose else inters
        for intervention in pbar:
            response = requests.get(intervention["document_url"]).json()["intervention"]
            interventions["interventions"].append(response)

        if save and slug_name:
            path = Path(save) / self.legislature_name / f"{slug_name}.json"
            path.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(interventions, f)

        return interventions

    @staticmethod
    async def process_response(session, url):
        async with session.get(url) as resp:
            action_item = await resp.json(content_type=None)
            return action_item["intervention"]

    async def async_fetch_interventions_of_deputy_2(
        self, dep_urls, slug_name=None, max_interventions=1000, verbose=True, save="./data/"
    ):
        # Initialize an empty list to store the interventions
        interventions = {
            "start": dep_urls["start"],
            "end": dep_urls["end"],
            "last_result": dep_urls["last_result"],
            "interventions": [],
        }

        inters = (
            dep_urls["results"][:max_interventions] if max_interventions else dep_urls["results"]
        )
        urls = [intervention["document_url"] for intervention in inters]
        pbar = tqdm(urls, leave=False) if verbose else urls

        async with aiohttp.ClientSession() as session:
            # Fetch the interventions from the API
            tasks = []
            for url in pbar:
                tmp = asyncio.ensure_future(self.process_response(session, url))
                tasks.append(tmp)

            interventions["interventions"] = await asyncio.gather(*tasks)

        if save and slug_name:
            path = Path(save) / self.legislature_name
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"{slug_name}.json", "w") as f:
                json.dump(interventions, f)

        return interventions

    def fetch_all_interventions(
        self,
        deputies_list_file="./data/interventions_urls_by_deputies.json",
        max_interventions=1000,
        verbose=[True, True],
        save="./data/",
    ):
        """Fetches all interventions from the API.

        Args:
            deputies_list_file: The file path to the interventions URLs by deputies.
                Default: "./data/interventions_urls_by_deputies.json".
            max_interventions: The maximum number of interventions to fetch.
                Default: 1000.
            verbose: Whether to display a progress bar, both level.
                Default: [True, True].
            save: The file path to save the results to.
                Default: "./data/interventions_by_deputies.json".

        Returns:
            A dictionary where the keys are deputy names and the values are lists of interventions.
        """
        # Get the URLs for all interventions
        with open(deputies_list_file, "r") as f:
            deputies_list = json.load(f)

        # Fetch the interventions from the API
        # deputies_interventions = {}

        pbar = tqdm(deputies_list, leave=False) if verbose[0] else deputies_list
        for dep in pbar:
            urls = deputies_list[dep]
            name = self.search_parlementaires(dep)[0][0]["nom"]
            slug_name = re.sub(" ", "+", unidecode.unidecode(name.lower()))
            self.fetch_interventions_of_deputy(
                urls,
                slug_name=slug_name,
                max_interventions=max_interventions,
                verbose=verbose[1],
                save=save,
            )
            # deputies_interventions[dep] = self.fetch_interventions_of_deputy(
            # urls, max_interventions=max_interventions, verbose=verbose[1], save=save)

        # # Save the results to a file if save path is provided
        # if save:
        #     path = Path(save)
        #     path_list = path.parts
        #     path_list.insert(-1, self.legislature_name)
        #     save_path = Path('').joinpath(*path_list)
        #     with open(save_path, 'w') as f:
        #         json.dump(deputies_interventions, f)
        #         self.interventions_file = save_path

        # return deputies_interventions

    # def interventions(self, dep_name):
    #     name = self.search_parlementaires(dep_name)[0][0]["nom"]
    #     name_pattern = re.sub(" ", "+", unidecode.unidecode(name.lower()))
    #     dep_intervention = []
    #     url = f"{self.base_url}/recherche?object_name=
    # Intervention&tag=parlementaire%3D{name_pattern}&sort=1"
    #     source = request.urlopen(url).read()
    #     page = bs4.BeautifulSoup(source, "lxml")
    #     for x in page.find_all("p", {"class": "content"}):
    #         dep_intervention += x

    #     return dep_intervention

    def session_title(self, dep_name):
        """Get the session title during a deputy intervention."""
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "+", unidecode.unidecode(name.lower()))
        title_session = []
        url = f"{self.base_url}/recherche?object_name=Intervention"
        url += f"&tag=parlementaire%3D{name_pattern}&sort=1"
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for title in page.find_all("h4"):
            title_session.append(title.get_text())

        return title_session

    def liste_mots(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "-", unidecode.unidecode(name.lower()))
        mots_dep = []
        url = f"{self.base_url}/{name_pattern}/tags"
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all("span", {"class": "tag_level_4"}):
            mots_dep.append(re.sub("\n", "", x.get_text()))

        return mots_dep
