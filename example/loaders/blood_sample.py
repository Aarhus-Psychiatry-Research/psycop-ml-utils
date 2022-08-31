import psycopmlutils.loaders.raw as raw_loaders

if __name__ == "__main__":
    df = raw_loaders.load_lab_results.LoadLabResults.hdl(n=100)

    pass
