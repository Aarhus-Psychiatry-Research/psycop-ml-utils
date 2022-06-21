
# Installation

## For use
```
pip install git+https://github.com/Aarhus-Psychiatry-Research/psycop-ml-utils.git
```
## For development
To install psycop-ml-utils for development simply run:
```bash
pip install -e .
```
The `-e` flag marks the install as editable, "overwriting" the package as you edit the source files.

For development you will also need to install development requirements as well as set up pre-commit hooks, which will format the code before pushing:

```bash
# install development requirements
pip install -r requirements.txt
# set up pre-comit
pre-commit install
```