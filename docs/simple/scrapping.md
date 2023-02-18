# Getting the data

For this project, we used data made openly available by [assemblee-nationale.fr/](https://www.assemblee-nationale.fr/).

The site [nosdeputes.fr](https://www.nosdeputes.fr) wrote a API to simplify the acces to the data.
We used this website to fetch the textual data used in the project.


## Wrapper function

!!! info "Wrapper function"
    We defined a wrapper function to simplify the use of the API. It fetches all the urls, interventions and the deputies information for a given legislature.

    It is used to fetch the data in the [notebook](../notebooks/scrapping.ipynb).

::: nlp_assemblee.api.fetch_data_for_legislature


## CPCApi

!!! info "API class"
    The wrapper function uses an API class inspired by the [GitHub of nosdeputes.fr](https://github.com/regardscitoyens/cpc-api).

::: nlp_assemblee.api.CPCApi
