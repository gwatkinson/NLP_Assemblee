# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from cpc_api.api import CPCApi
# sometimes, just use cpc_api.api when it works
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
    names = deputies_of_group(group, n_deputies)
    print(names)
    interventions = []
    for name in names:
        print(name)
        interventions += [[group, name, apidep.interventions2(name)]]
    return interventions

def stockintervention(groupe):
    interventions_group = []
    nbdep = deputies_df.groupby('groupe_sigle')['nom'].count()[str(groupe)]
    print(nbdep)
    interventions_group += interventions_of_group(groupe, nbdep)
    interventions_df = pd.DataFrame(
        interventions_group,
        columns=["groupe", "nom", "interventions"]
        )
    
    return interventions_df