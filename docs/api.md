# Scrapping the data

For this project, we used data made openly available by [assemblee-nationale.fr/](https://www.assemblee-nationale.fr/).

The site [nosdeputes.fr](https://www.nosdeputes.fr) wrote a API to simplify the acces to the data.
We used this website to fetch the textual data used in the project.

## CPCApi

We defined a class inspired by the [GitHub of nosdeputes.fr](https://github.com/regardscitoyens/cpc-api).

We then used it to fetch the data in the [notebook](https://github.com/gwatkinson/NLP_Assemblee/blob/main/notebooks/scrapping.ipynb).

::: nlp_assemblee.scrapping.api.CPCApi
