EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}


def add_event_weight(df):
    df = df.copy()
    df["event_weight"] = df["event"].map(EVENT_WEIGHTS).fillna(0)
    return df


def get_user_features(df):
    df = add_event_weight(df)
    return df.groupby("visitorid").agg(
        total_events=("event", "size"),
        unique_items=("itemid", "nunique"),
        total_weight=("event_weight", "sum"),
    ).reset_index()


def get_item_features(df):
    df = add_event_weight(df)
    return df.groupby("itemid").agg(
        total_events=("event", "size"),
        unique_users=("visitorid", "nunique"),
        total_weight=("event_weight", "sum"),
    ).reset_index()
