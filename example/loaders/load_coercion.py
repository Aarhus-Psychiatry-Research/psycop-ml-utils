from psycopmlutils.loaders.raw.load_coercion import LoadCoercion

if __name__ == "__main__":
    df = LoadCoercion.coercion_duration(n=100)
    farlighed = LoadCoercion.farlighed(n=20)
    bælte = LoadCoercion.bælte(n=100)

    pass
