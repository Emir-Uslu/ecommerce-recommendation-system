EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}


def recommend_popular_items(df, n=10):
    df = df.copy()
    df["score"] = df["event"].map(EVENT_WEIGHTS).fillna(0)
    return (
        df.groupby("itemid", as_index=False)["score"]
        .sum()
        .sort_values("score", ascending=False)
        .head(n)
    )
