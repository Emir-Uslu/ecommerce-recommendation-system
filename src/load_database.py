import pandas as pd
from sqlalchemy import create_engine
from getpass import getpass


user = "postgres"
password = getpass("PostgreSQL password: ")
host = "localhost"
port = "5432"
database = "ecommerce_recommendation"

engine = create_engine(
    f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"
)

df = pd.read_csv(
    "data/raw/events.csv"
)

print("Rows:", len(df))
print("Loading data to PostgreSQL...")

df.to_sql(
    "events",
    engine,
    if_exists="replace",
    index=False,
    chunksize=5000,
    method="multi"
)

print("Data loaded successfully.")