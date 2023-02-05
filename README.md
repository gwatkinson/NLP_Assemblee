# NLP and French Politics


[![GitHub](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/gwatkinson/NLP_Assemblee/) [![Website shields.io](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://gwatkinson.github.io/NLP_Assemblee/) [![GitHub license](https://img.shields.io/github/license/gwatkinson/NLP_Assemblee.svg)](https://github.com/gwatkinson/NLP_Assemblee/blob/main/LICENSE) [![GitHub branches](https://badgen.net/github/branches/gwatkinson/NLP_Assemblee)](https://github.com/gwatkinson/NLP_Assemblee/branches)

This repository contains the NLP project for the Deep Learning course of the MVA.

This is a group project with Gabriel Watkinson and Jéremie Stym-Popper.

<!-- toc -->

- [Problematic](#problematic)
- [Documentation](#documentation)
- [Replicating the Results](#replicating-the-results)
    * [Clone the Repository](#clone-the-repository)
    * [Create a Conda Enviromnent](#create-a-conda-enviromnent)
- [Development dependencies](#development-dependencies)
    * [Formatters, Linters and Documentation](#formatters-linters-and-documentation)
    * [Pre-commit](#pre-commit)
    * [Git-Flow](#git-flow)

<!-- tocstop -->

## Problematic

This project aims to recreate a map of the French Assemblée Nationale, by classifying the deputees' political opinions on a range of subjects.

## Documentation

We have a [https://gwatkinson.github.io/NLP_Assemblee/](https://gwatkinson.github.io/NLP_Assemblee/), hosted with GitHub Pages, with the documentation of the project.

## Replicating the Results

To install the dependencies needed for running the code locally, follow the next steps:

### Clone the Repository

Clone the repository with the command:

```bash
git clone https://github.com/gwatkinson/NLP_Assemblee.git
```

Or download the zip from the [github](https://github.com/gwatkinson/NLP_Assemblee) and unzip it where you want.

Then, move into it:

```bash
cd NLP_Assemblee
```

### Create a Conda Enviromnent

Make sure you have conda or [miniconda](https://docs.conda.io/en/latest/miniconda.html) installed (see [here](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html) for instructions to install conda).

To create the conda enviromnent named `YOURENV`, choose your platform and use the following commands:

Create conda enviromnent for Linux-64:

```bash
conda create --name YOURENV --file env_file/conda-linux-64.lock
```

Create conda enviromnent for Windows-64:

```bash
conda create --name YOURENV --file env_file/conda-win-64.lock
```

This can take a while since there are many packages that are quite big (namely pytorch).

You can then activate the enviromnent:

```bash
conda activate YOURENV
```

To generate the lock file from `environment.yml`, run:

```bash
conda-lock -k explicit --conda mamba
```

## Development dependencies

We used [pre-commit](https://pre-commit.com/) to run some formatters and other hooks before each commit.
And we used [git-flow](https://github.com/nvie/gitflow) to manage the different branches.

### Formatters, Linters and Documentation

To install the development dependencies with poetry, run:

```bash
poetry install
```

Or with pip:

```bash
pip install -e .
```

### Pre-commit

If you want to contribute, please install the pre-commit hooks (in the root folder with git and with the enviromnent activated):

```bash
pre-commit install
```

This installs hooks to /.git/hooks

and run it once against the code:

```bash
pre-commit run --all-files
```

This will run some formatters and other hooks before each commit.

### Git-Flow

This project tries to use the [gitflow](https://github.com/nvie/gitflow) workflow. It relies on multiple branches:

- main
- develop
- feature/*
- release/*
- hotfixe

To use it, please [install git-flow](https://skoch.github.io/Git-Workflow/), then initialize the project:

```bash
git flow init
```

select the default values for the names.

Articles to get started:

- [https://jeffkreeftmeijer.com/git-flow/](https://jeffkreeftmeijer.com/git-flow/)
- [https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)
