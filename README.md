# NLP and French politic

This repository contains the NLP project for the Deep Learning course of the MVA.


## Installation

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

### Poetry

This project uses poetry to manage the Python packages. So, before installing the dependancies, [install poetry](https://python-poetry.org/docs/#installation).

**On linux, MacOs, WSL:**
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

**On Windows:**
```Powershell
 (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
```

Then, make sure it is in your `$PATH`. You can check that by running :

```bash
poetry --version
```

If this command doesn't work, look at the [documentation](https://python-poetry.org/docs/#installation).

### Install the packages

Once poetry is installed, you can install the packages:
```
poetry install
```

This will install the exact versions specified in the `poetry.lock` file.

This will create an enviromnent and install the packages into it.
However, if you already have a venv activated, it will install them into it.
You can use both conda or basic Python environment.

## Contributing

This section describes some norms to follow while developping in this project.

### Pre-commit

If you want to contribute, please install the pre-commit hooks:
```
pre-commit install
```
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

Articles to get started:

* https://jeffkreeftmeijer.com/git-flow/
* https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow
