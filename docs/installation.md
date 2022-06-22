
# Installation
To get started using psycop-ml-utils simply install it using pip by running the following line in your terminal:

Install using your preferred package manager, e.g.:
`pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

or using peotry by running

`poetry add git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`


## For development
We use poetry for dependency management. To install poety following the instruction on their [website](https://python-poetry.org/docs/#osx--linux--bashonwindows-install-instructions).


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
