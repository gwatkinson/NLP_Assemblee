# NLP and French politic

This repository contains the NLP project for the Deep Learning course of the MVA.

This is a group project with Gabriel Watkinson and Jéremie Stym-Popper.

<!-- toc -->

- [Problematic](#problematic)
- [Contributing](#contributing)
    * [Clone the repository](#clone-the-repository)
    * [Create conda enviromnent](#create-conda-enviromnent)
    * [Pre-commit](#pre-commit)
    * [Git-Flow](#git-flow)

<!-- tocstop -->

## Problematic

This project aims to recreate a map of the French Assemblée Nationale, by classifying the deputees' political opinions on a range of subjects.


## Contributing

To install the dependencies needed for running the code locally, follow the next steps:

### Clone the repository

Clone the repository with the command:
```bash
git clone https://github.com/gwatkinson/NLP_Assemblee.git
```
Or download the zip from the [github](https://github.com/gwatkinson/NLP_Assemblee) and unzip it where you want.

Then, move into it:
```bash
cd NLP_Assemblee
```

### Create conda enviromnent

Make sure you have conda or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed (see [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) for instructions to install conda).

To create the conda enviromnent named `YOURENV`, choose your platform and use the following commands:

Create conda enviromnent for Linux-64:
```
conda create --name YOURENV --file env_file/conda-linux-64.lock
```

Create conda enviromnent for Windows-64:
```
conda create --name YOURENV --file env_file/conda-win-64.lock
```
This can take a while since there are many packages that are quite big (namely pytorch).

You can then activate the enviromnent:
```
conda activate YOURENV
```

To generate the lock file from `environment.yml`, run:
```
conda-lock -k explicit --conda mamba
```

### Pre-commit

If you want to contribute, please install the pre-commit hooks (in the root folder with git and with the enviromnent activated):
```
pre-commit install
```
This installs hooks to /.git/hooks

and run it once against the code:
```
pre-commit run --all-files
```

This will run some formatters and other hooks before each commit.

### Git-Flow

This project tries to use the [gitflow](https://github.com/nvie/gitflow) workflow. It relies on multiple branches:

* main
* develop
* feature/*
* release/*
* hotfixe

To use it, please [install git-flow](https://skoch.github.io/Git-Workflow/), then initialize the project:
```
git flow init
```
select the default values for the names.


Articles to get started:

* https://jeffkreeftmeijer.com/git-flow/
* https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
