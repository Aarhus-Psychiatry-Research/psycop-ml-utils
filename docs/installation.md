
# Installation
To get started using psycop-ml-utils simply install it using pip by running the following line in your terminal:

Install using your preferred package manager, e.g.:
`pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

or using peotry by running

`poetry add git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`


## For development
We use poetry for dependency management. 

Install poetry with the following command:

```
curl -sSL https://install.python-poetry.org | python3 -
```

Add to your shell configuration as described in the output, then run:
```
poetry config virtualenvs.in-project true
```

````{note}
You might get the error `command not found: poetry` in which case you need to export the path to poetry using:

```
export PATH="$HOME/.poetry/bin:$PATH"
```

````  
  

To make poetry save venvs in your project directories.



Clone the repo, move into it, then run `poetry install`. I.e.:

```bash
git clone https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git
cd psycop-ml-utils
poetry install
```

To increase the version:
`poetry version [patch|minor|major]` according to [semantic versioning](https://semver.org/).

Adding new as dependnecies:
`poetry add (--dev) [packagename]`

No need to update a `requirements.txt`. It's replace by `pyproject.toml`, and `poetry` manages it automatically.


## When using
Install using your preferred package manager, e.g.:
`pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

or

`poetry add git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`
