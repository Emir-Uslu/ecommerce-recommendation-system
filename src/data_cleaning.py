import pandas as pd


def load_events(path):
    return pd.read_csv(path)


def clean_events(df):
    df = df.copy()
    df = df.drop_duplicates()
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.day_name()
    return df


def get_event_summary(df):
    result = df["event"].value_counts().rename_axis("event").reset_index(name="count")
    result["share_pct"] = (result["count"] / result["count"].sum() * 100).round(2)
    return result
