import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5,
}


def create_user_item_matrix(df):
    df = df.copy()
    df["weight"] = df["event"].map(EVENT_WEIGHTS).fillna(0)
    return df.pivot_table(
        index="visitorid",
        columns="itemid",
        values="weight",
        aggfunc="sum",
        fill_value=0,
    )


def create_item_similarity(matrix):
    item_user = matrix.T
    similarity = cosine_similarity(item_user)
    return pd.DataFrame(similarity, index=item_user.index, columns=item_user.index)


def recommend_similar_items(item_id, similarity_df, n=10):
    if item_id not in similarity_df.index:
        return []
    scores = similarity_df.loc[item_id].drop(item_id).sort_values(ascending=False)
    return scores.head(n).index.tolist()
