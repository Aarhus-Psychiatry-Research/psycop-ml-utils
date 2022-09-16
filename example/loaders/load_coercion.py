"""Testing loading of coercion functions."""

# pylint: disable=non-ascii-name

from psycopmlutils.loaders.raw.load_coercion import *

if __name__ == "__main__":
    df = coercion_duration(n=100)
    farlighed = farlighed(n=20)
    bælte = bælte(n=100)
