from prefect import flow, task
import pandas as pd

@task
def extract():
    return pd.read_csv("titanic.csv")

@task
def transform(df):
    df = df.dropna(subset=["age"])
    return df

@task
def load(df):
    df.to_csv("titanic_clean.csv", index=False)

@flow
def etl_flow():
    data = extract()
    clean = transform(data)
    load(clean)

if __name__ == "__main__":
    etl_flow()
