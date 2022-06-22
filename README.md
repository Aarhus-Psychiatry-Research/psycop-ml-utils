<a href="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils"><img src="https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/blob/main/docs/icon.png?raw=true" width="200" align="right" /></a>
# PSYCOP Machine Learning Utilities

![python versions](https://img.shields.io/badge/Python-%3E=3.7-blue)
[![Code style: black](https://img.shields.io/badge/Code%20Style-Black-black)](https://black.readthedocs.io/en/stable/the_black_code_style/current_style.html)
[![github actions pytest](https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/actions/workflows/pytest.yml/badge.svg)](https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/actions)
[![github actions docs](https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/actions/workflows/documentation.yml/badge.svg)](https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/)
![coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/martbern/d6c40a5b5a3169c079e8b8f778b8e517/raw/badge-psycop-ml-utils-pytest-coverage.json)

# Installation
## For development
We use poetry for dependency management. Clone the repo, move into it, then run `poetry install`. I.e.:

```bash
git clone https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git
cd psycop-ml-utils
poetry install
```

To increase the version:
`poetry version [patch|minor|major]` according to [semantic versioning](https://semver.org/).

Adding new as dependencies:
`poetry add (--dev) [packagename]`

No need to update a `requirements.txt`. It's replace by `pyproject.toml`, and `poetry` manages it automatically.


## When using
Install using your preferred package manager, e.g.:
`pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

or

`poetry add git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git`

# Usage
- [ ] Update examples as API matures


## 🔧 Installation
To get started using psycop-ml-utils simply install it using pip by running the following line in your terminal:

```
pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git
```

For more detailed instructions on installation, see the [installation instructions](https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/installation).


## 📖 Documentation

| Documentation              |                                                                             |
| -------------------------- | --------------------------------------------------------------------------- |
| 📚 **[Usage Guides]**       | Guides and instructions on how the package and its features.            |
| 📰 **[News and changelog]** | New additions, changes and version history.                                 |
| 🎛 **[API References]**     | The detailed reference for psycop-ml-utils's API. Including function documentation |
| 🙋 **[FAQ]**                | Frequently asked question                                |

[usage guides]: https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/introduction.html
[api references]: https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/
[Augmenters]: https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/augmenters.html
[Demo]: https://share.streamlit.io/Aarhus-Psychiatry-Research/psycop-ml-utils/dev/streamlit.py
[News and changelog]: https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/news.html
[FAQ]: https://Aarhus-Psychiatry-Research.github.io/psycop-ml-utils/faq.html

## 💬 Where to ask questions

| Type                           |                        |
| ------------------------------ | ---------------------- |
| 🚨 **Bug Reports**              | [GitHub Issue Tracker] |
| 🎁 **Feature Requests & Ideas** | [GitHub Issue Tracker] |
| 👩‍💻 **Usage Questions**          | [GitHub Discussions]   |
| 🗯 **General Discussion**       | [GitHub Discussions]   |

[github issue tracker]: https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/issues
[github discussions]: https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils/discussions


