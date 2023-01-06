# -*- coding: utf-8 -*-

"""A Python wrapper for the Regards Citoyens nosdeputes.fr and nossenateurs.fr
APIs.

These APIs provide information about French parliamentary deputies, such
as their names, parties, and parliamentary interventions.
"""

import requests
import warnings

import re
import bs4
from urllib import request
import unidecode

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

    format = "json"

    def __init__(self, ptype="depute", legislature=None):
        """Initializes the API object.

        Args:
            ptype (str, optional): The type of parliamentarian to retrieve data for.
                Valid values are "depute" (deputy) or "senateur" (senator).
                Default: "depute"
            legislature (str, optional): The legislature to retrieve data for.
                Valid values are "2007-2012" or "2012-2017", or None for current legislature.
                Default: None

        Raises:
            AssertionError: If ptype is not "depute" or "senateur", or if legislature is not
                "2007-2012", "2012-2017", or None.
        """
        assert ptype in ["depute", "senateur"]
        assert legislature in ["2007-2012", "2012-2017", None]
        self.legislature = legislature
        self.ptype = ptype
        self.ptype_plural = ptype + "s"
        self.base_url = "https://%s.nos%s.fr" % (legislature or "www", self.ptype_plural)

    def synthese(self, month=None):
        """month format: YYYYMM."""
        if month is None and self.legislature == "2012-2017":
            raise AssertionError(
                "Global Synthesis on legislature does not work,"
                + "see https://github.com/regardscitoyens/nosdeputes.fr/issues/69"
            )

        if month is None:
            month = "data"

        url = "%s/synthese/%s/%s" % (self.base_url, month, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def parlementaire(self, slug_name):
        url = "%s/%s/%s" % (self.base_url, slug_name, self.format)
        return requests.get(url).json()[self.ptype]

    def picture(self, slug_name, pixels="60"):
        return requests.get(self.picture_url(slug_name, pixels=pixels))

    def picture_url(self, slug_name, pixels="60"):
        return "%s/%s/photo/%s/%s" % (self.base_url, self.ptype, slug_name, pixels)

    def search(self, q, page=1):
        # XXX : the response with json format is not a valid json :'(
        # Temporary return csv raw data
        url = "%s/recherche/%s?page=%s&format=%s" % (self.base_url, q, page, "csv")
        return requests.get(url).content

    @memoize
    def parlementaires(self, active=None):
        if active is None:
            url = "%s/%s/%s" % (self.base_url, self.ptype_plural, self.format)
        else:
            url = "%s/%s/enmandat/%s" % (self.base_url, self.ptype_plural, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def search_parlementaires(self, q, field="nom", limit=5):
        return extractBests(
            q,
            self.parlementaires(),
            processor=lambda x: x[field] if type(x) == dict else x,
            limit=limit,
        )

    def interventions(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "+", unidecode.unidecode(name.lower()))
        dep_intervention = []
        url = "https://www.nosdeputes.fr/recherche?object_name=Intervention&tag=parlementaire%3D{0}&sort=1".format(
            name_pattern
        )
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all("p", {"class": "content"}):
            dep_intervention += x

        return dep_intervention

    def session_title(self, dep_name):
        """Get the session title during a deputy intervention."""
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "+", unidecode.unidecode(name.lower()))
        title_session = []
        url = "https://www.nosdeputes.fr/recherche?object_name=Intervention&tag=parlementaire%3D{0}&sort=1".format(
            name_pattern
        )
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for title in page.find_all("h4"):
            title_session.append(title.get_text())

        return title_session

    def liste_mots(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "-", unidecode.unidecode(name.lower()))
        mots_dep = []
        url = "https://www.nosdeputes.fr/{0}/tags".format(name_pattern)
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all("span", {"class": "tag_level_4"}):
            mots_dep.append(re.sub("\n", "", x.get_text()))

        return mots_dep
