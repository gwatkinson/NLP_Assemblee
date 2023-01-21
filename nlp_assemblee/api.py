# -*- coding: utf-8 -*-

"""A Python wrapper for the Regards Citoyens nosdeputes.fr and nossenateurs.fr
APIs.

These APIs provide information about French parliamentary deputies, such
as their names, parties, and parliamentary interventions.
"""

import asyncio
import json
import re
import time
import warnings
from glob import glob
from pathlib import Path

# from grequests import async
import aiohttp
import pandas as pd
import requests
import unidecode
from tqdm.autonotebook import tqdm

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from fuzzywuzzy.process import extractBests


def memoize(f):
    """Decorator to memoize a function.

    Args:
        f (function): The function to memoize

    Returns:
        aux (function): The memoized function
    """
    cache = {}

    def aux(*args, **kargs):
        k = (args, tuple(sorted(kargs.items())))
        if k not in cache:
            cache[k] = f(*args, **kargs)
        return cache[k]

    return aux


class CPCApi(object):
    """The CPCApi class provides an interface for interacting with
    nosdeputes.fr's API. It allows users to retrieve information about
    parliamentarians, as well as the interventions they have made.

    The class provides asynchronous methods for fetching data, as well as methods for processing
    and saving the data. Additionally, the class contains methods for handling pagination and
    error handling. The class also provides a way to save the result of the request to a folder
    or file.

    The class uses aiohttp, asyncio and json library to make the request, parse the json and save
    the result to a file.
    """

    def __init__(self, ptype="depute", legislature="2017-2022", format="json"):
        """Initializes the API object.

        Args:
            ptype (str): The type of parliamentarian to retrieve data for.
                Valid values are "depute" (deputy) or "senateur" (senator).
            legislature (str): The legislature to retrieve data for.

                Valid values are "2007-2012" or "2012-2017", "2017-2022",
                or None for current legislature.

                Note that older legislatures have problems for some processing later on.
            format (str): The format to retrieve data in.
                Valid values are "json" or "xml".

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

    @memoize
    def parlementaires(self, active=None):
        """Retrieves a list of parliamentaries.

        Args:
            active (bool): Whether to retrieve active parliamentaries or not.

        Returns:
            parlementaires_list (list): A list of parliamentaries. This also creates the attribute
                self.parlementaires_list.

        Creates:
            self.parlementaires_list : A list of all parliamentaries
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

    def search_parlementaires(self, q, field="nom", limit=5):
        """Fuzzy searches for a parliamentarian in the deputies list.

        Args:
            q (str): The query to search for.
            field (str): The field of the parliamentarian data to search in.
            limit (int): The maximum number of results to return.

        Returns:
            name (str): The exact name of the parliamentarian in the list.
        """
        # Search for parliamentarians and return the best matches
        name = extractBests(
            q,
            self.parlementaires_list,
            processor=lambda x: x[field] if type(x) == dict else x,
            limit=limit,
        )
        return name

    def get_deputies_df(self):
        """Retrieves a DataFrame of deputies information from the API.

        Returns:
            deputies_df (pandas.DataFrame): A DataFrame containing information about deputies.

        Creates:
            self.deputies_df : A DataFrame containing information about deputies
            self.groups : groups of the deputies
            self.deputies : Deputies information
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

        return deputies_df

    @staticmethod
    async def urls_response(session, url, count_param, page_param):
        """Retrieves the json response from a url.

        Args:
            session (aiohttp.ClientSession): The session to use for the request
            url (str): The url to retrieve the response from
            count_param (str): The parameter for the number of results in the url
            page_param (str): The parameter for the page number in the url

        Returns:
            response (dict): The json response from the url.
                    Returns an empty dict if there is an error.
                    The dict contains keys (start, end, last_result, results)
        """
        async with session.get(url + count_param + page_param) as resp:
            try:
                tmp = await resp.json(content_type=None)
            except Exception as e:
                print(e)
                return {
                    "start": 1,
                    "end": 0,
                    "last_result": 0,
                    "results": [],
                }
        return tmp

    async def async_get_deputy_interventions_urls(
        self,
        dep_name,
        all_pages=False,
        max_interventions=1000,
        count=500,
        last_int=500,
        sort=0,
        page=1,
    ):
        """Retrieves the urls of the interventions of a parliamentarian.

        Args:
            dep_name (str): The name of the parliamentarian.
            all_pages (bool, optional): Whether to retrieve all pages of results.
            max_interventions (int, optional): Maximum number of interventions to retrieve.
            count (int, optional): The number of interventions to retrieve per page.
            last_int (int, optional): The last intervention number to retrieve.
            sort (int, optional): The sort order of the interventions, by pertinence or by date.
            page (int, optional): The page of results to retrieve.

        Returns:
            response (dict): A json object containing the links to the interventions.
        """
        if all_pages or max_interventions:
            page = 1

        # Remove accents and special characters from the name and replace spaces with +
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        slug_name = re.sub(" ", "+", unidecode.unidecode(name.lower()))

        # Build the URL for the search
        url = f"{self.base_url}/recherche?object_name=Intervention&format={self.format}"
        url += f"&tag=parlementaire%3D{slug_name}&sort={sort}"
        page_param = f"&page={page}"
        count_param = f"&count={count}"

        if last_int is None:
            tick = requests.get(url + "&count=0&page=1").json()["last_result"]

            if all_pages:
                last_int = tick
            else:
                last_int = min(max_interventions, tick)

        response = {"start": 1, "end": 0, "last_result": last_int, "results": []}

        # Retrieve the search results
        async with aiohttp.ClientSession(trust_env=True) as session:
            tasks = []
            for page in range(1, last_int // count + 2):
                page_param = f"&page={page}"
                tasks.append(
                    asyncio.ensure_future(self.urls_response(session, url, count_param, page_param))
                )
            responses = await asyncio.gather(*tasks)

            for tmp in responses:
                response["results"] += tmp["results"]
                response["end"] = tmp["end"]

            response["dep_name"] = dep_name
            response["slug_name"] = slug_name
            response["legislature"] = self.legislature_name

            return response

    async def async_get_all_interventions_urls(
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

        Args:
            all_pages (bool, optional): Whether to retrieve all pages of results or just the
                first page.
            max_interventions (int, optional): Maximum number of interventions to retrieve.
            sort (int, optional): An integer representing the sorting method to use.
            count (int, optional): The number of results to retrieve per page.
            object_name (str, optional): The name of the object to retrieve.
            page (int, optional): The page number to start at.
            verbose (bool, optional): Whether to display a progress bar.
            save (str, optional): The file path to save the results to.

        Returns:
            deputies_list (dict): A dictionary where the keys are deputy names and the values
                are lists of URLs for their interventions.
        """
        deputies_list = {}

        async with aiohttp.ClientSession(trust_env=True) as session:
            tasks = []
            for dep in self.deputies:
                tmp = asyncio.ensure_future(
                    self.async_get_deputy_interventions_urls(
                        dep_name=dep,
                        session=session,
                        count=count,
                        object_name=object_name,
                        sort=sort,
                        page=page,
                        all_pages=all_pages,
                        max_interventions=max_interventions,
                    )
                )
                tasks.append(tmp)

            responses = await asyncio.gather(*tasks)

        for dep, response in zip(self.deputies, responses):
            deputies_list[dep] = response

        # Save the results to a file if save path is provided
        if save:
            path = Path(save)
            path_list = list(path.parts)
            path_list.insert(2, self.legislature_name)
            save_path = Path("").joinpath(*path_list[:-1])
            save_path.mkdir(parents=True, exist_ok=True)
            with open(save_path / path_list[-1], "w") as f:
                json.dump(deputies_list, f)
                self.deputies_list_file = str(save_path / path_list[-1])

        return deputies_list

    @staticmethod
    async def process_response(session, url):
        """Given a url, this function calls the url and returns the json object
        associated with the url.

        Args:
            session : The aiohttp session object
            url : The url to call

        Returns:
            intervention (dict): json object containing the intervention details
        """
        async with session.get(url) as resp:
            action_item = await resp.json(content_type=None)
            return action_item["intervention"]

    async def async_fetch_interventions_of_deputy(
        self, dep_urls, slug_name=None, max_interventions=1000, verbose=True, save="./data/"
    ):
        """Retrieves the interventions of a parliamentarian.

        Args:
            dep_urls (dict): A dictionary containing the links to the interventions.
            slug_name (str): The slug name of the parliamentarian.
            max_interventions (int): The maximum number of interventions to retrieve.
            verbose (bool): Whether to display a progress bar.
            save (str):The file path to save the results to.

        Returns:
            interventions (dict): A dictionary containing the interventions.
        """
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

        async with aiohttp.ClientSession(trust_env=True) as session:
            # Fetch the interventions from the API
            tasks = []
            for url in pbar:
                tmp = asyncio.ensure_future(self.process_response(session, url))
                tasks.append(tmp)

            interventions["interventions"] = await asyncio.gather(*tasks)

        if save and slug_name:
            path = Path(save) / self.legislature_name / "interventions"
            path.mkdir(parents=True, exist_ok=True)
            with open(path / f"{slug_name}.json", "w") as f:
                json.dump(interventions, f)

        return interventions

    async def async_fetch_all_interventions(self, urls, save="./data/", max_interventions=1000):
        """Retrieves all interventions for each deputy and saves them to a file
        for each deputy.

        Args:
            urls (dict): A dictionary where the keys are deputy names and the values are lists of
                URLs for their interventions.
            save (str): The file path to save the results to.
            max_interventions (int): The maximum number of interventions to fetch for each deputy.

        Returns:
            interventions_dict (dict): A dictionary where the keys are the deputy names and the
                values are lists of their interventions.
        """
        deps = self.deputies
        slugs = [self.deputies_df[self.deputies_df["nom"] == dep]["slug"].values[0] for dep in deps]
        files = glob(f"{save}/{self.legislature_name}/interventions/*.json")
        files = [file.split("/")[-1].split(".")[0] for file in files]

        idx = [i for i, slug in enumerate(slugs) if slug not in files]
        deps = [deps[i] for i in idx]
        slugs = [slugs[i] for i in idx]

        print(f"Fetching {len(idx)} deputies' interventions... Instead of {len(self.deputies)}")

        interventions_dict = {}

        for dep, slug in tqdm(zip(deps, slugs), total=len(idx)):
            try:
                interventions_dict[dep] = await self.async_fetch_interventions_of_deputy(
                    dep_urls=urls[dep],
                    slug_name=slug,
                    max_interventions=max_interventions,
                    verbose=False,
                    save=save,
                )
            except Exception as e:
                print(dep, e)
                time.sleep(60)

        return interventions_dict


async def fetch_data_for_legislature(
    legislature="2017-2022", root_dir="./data/", max_interventions=500
):
    """Fetch data for a given legislature, including interventions and deputies
    data.

    Args:
        legislature (str): the legislature to fetch data for.
        root_dir (str): the root directory to save data.
        max_interventions (int): the maximum number of interventions to fetch.

    Returns:
        uresults (tuple): containing the urls of the interventions, the interventions data
            and the deputies data
    """
    api = CPCApi(legislature=legislature)

    deps_df = api.deputies_df

    urls = await api.async_get_all_interventions_urls(
        all_pages=False,
        max_interventions=max_interventions,
        sort=0,
        count=500,
        object_name="Intervention",
        page=1,
        verbose=True,
        save=f"{root_dir}/interventions_urls_by_deputies.json",
    )

    interventions = await api.async_fetch_all_interventions(
        urls, save=root_dir, max_interventions=max_interventions
    )

    return urls, interventions, deps_df
