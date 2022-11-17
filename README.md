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

### Create conda enviromnent

Create conda enviromnent for Linux-64:
```
conda create --name YOURENV --file conda-linux-64.lock
```

Create conda enviromnent for Windows-64:
```
conda create --name YOURENV --file conda-win-64.lock
```

Create conda enviromnent for Mac:
```
conda create --name YOURENV --file conda-osx-arm64.lock
```

You can then activate the enviromnent:
```
conda activate YOURENV
```

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
