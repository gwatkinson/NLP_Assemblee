# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from src.cpc_api.api import CPCApi
# sometimes, use src.cpc_api.api or just cpc_api.api
# ignore the warning

# load deputies dataframe
apidep = CPCApi()
deputies_json = apidep.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

# get list of all *groupes parlementaires*
groupes = deputies_df["groupe_sigle"].unique()

# Intermediary functions
def deputies_of_group(group, n_deputies):
    all_names = deputies_df[deputies_df["groupe_sigle"] == group]["nom"]
    return all_names[:n_deputies]


def interventions_of_group(group, n_deputies=15):
    """
    Return a list of 50 interventions by deputy of one group.
    Presented: acronym of the group, name of deputy, 50 interventions (list of str)
    Args: group (str) Acronym of the group
        n_deputies: (int) max number of selected deputies in the group. 
        no problem if it is larger than the number of deputies
    
    Return: a 3 column dataframe [group, deputy's name, 50 interventions]
    """
    names = deputies_of_group(group, n_deputies)
    print(names)
    interventions = []
    for name in names:
        print(name)
        interventions += [[group, name, apidep.interventions2(name)]]
    return interventions

def stockintervention(groupe):
    """
    Same as above but without limit of the number of deputies
    (takes them all). Could be a long process

    Return: a 3 column dataframe [group, deputy's name, 50 interventions]
    """
    interventions_group = []
    nbdep = deputies_df.groupby('groupe_sigle')['nom'].count()[str(groupe)]
    print(nbdep)
    interventions_group += interventions_of_group(groupe, nbdep)
    interventions_df = pd.DataFrame(
        interventions_group,
        columns=["groupe", "nom", "interventions"]
        )
    
    return interventions_df


#### This part to create a huge dataframe with all deputies, all groups
# and 50 interventions for each 
# WARNING : take a long time, do not run it unless necessary

# interventions_df = pd.DataFrame(columns=["groupe", "nom", "interventions"])

# for groupe in groupes:
#     interventions_df = pd.concat([interventions_df, stockintervention(groupe)], ignore_index=True)

# path = "mypath"
# interventions_df.to_csv(path + "\depinter.csv",index=False)

