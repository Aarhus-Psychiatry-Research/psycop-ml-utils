"""Testing loading of coercion functions."""

# pylint: disable=non-ascii-name

import psycopmlutils.loaders.raw.load_coercion as c

if __name__ == "__main__":
    df = c.load_corcioncoercion_duration(n=100)
    farlighed = c.farlighed(n=20)
    bælte = c.bælte(n=100)
