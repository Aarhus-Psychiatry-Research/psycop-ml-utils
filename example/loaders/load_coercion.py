from psycopmlutils.loaders.raw.load_coercion import LoadCoercion

if __name__ == "__main__":
    df = LoadCoercion.coercion(n=100)
    farlighed = LoadCoercion.coercion_farlighed(n=20)
    bælte = LoadCoercion.bælte(n=100)

    pass
