import pandas as pd
import numpy as np
import streamlit as st

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


DATA_PATH = "data/raw/events.csv"

EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5
}

MIN_USER_EVENTS = 7
MIN_ITEM_EVENTS = 15
N_NEIGHBORS = 25
SEED_ITEMS = 5


st.set_page_config(
    page_title="E-Commerce Recommendation System",
    layout="wide"
)


@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        usecols=[
            "timestamp",
            "visitorid",
            "event",
            "itemid",
            "transactionid"
        ]
    )

    df = df.drop_duplicates()

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="ms"
    )

    df["weight"] = df["event"].map(
        EVENT_WEIGHTS
    )

    return df


@st.cache_resource
def build_model(df):

    user_counts = df["visitorid"].value_counts()

    active_users = user_counts[
        user_counts >= MIN_USER_EVENTS
    ].index

    filtered = df[
        df["visitorid"].isin(active_users)
    ].copy()

    item_counts = filtered["itemid"].value_counts()

    active_items = item_counts[
        item_counts >= MIN_ITEM_EVENTS
    ].index

    filtered = filtered[
        filtered["itemid"].isin(active_items)
    ].copy()

    interactions = (
        filtered.groupby(
            ["visitorid", "itemid"],
            as_index=False
        )["weight"]
        .sum()
    )

    user_ids = interactions[
        "visitorid"
    ].unique()

    item_ids = interactions[
        "itemid"
    ].unique()

    user_to_idx = {
        user_id: idx
        for idx, user_id in enumerate(user_ids)
    }

    item_to_idx = {
        item_id: idx
        for idx, item_id in enumerate(item_ids)
    }

    idx_to_item = {
        idx: item_id
        for item_id, idx in item_to_idx.items()
    }

    rows = interactions[
        "visitorid"
    ].map(user_to_idx)

    cols = interactions[
        "itemid"
    ].map(item_to_idx)

    values = interactions[
        "weight"
    ]

    matrix = csr_matrix(
        (
            values,
            (rows, cols)
        ),
        shape=(
            len(user_ids),
            len(item_ids)
        )
    )

    item_matrix = matrix.T

    knn = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=N_NEIGHBORS,
        n_jobs=-1
    )

    knn.fit(item_matrix)

    popularity = (
        filtered.groupby("itemid")["weight"]
        .sum()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    return {
        "filtered": filtered,
        "matrix": matrix,
        "item_matrix": item_matrix,
        "knn": knn,
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_item": idx_to_item,
        "popularity": popularity
    }


def recommend(user_id, model, k=10):

    if user_id not in model["user_to_idx"]:
        return model["popularity"][:k]

    user_idx = model[
        "user_to_idx"
    ][user_id]

    user_vector = model[
        "matrix"
    ].getrow(user_idx)

    interacted_indices = user_vector.indices
    interacted_weights = user_vector.data

    seed_order = np.argsort(
        interacted_weights
    )[::-1][:SEED_ITEMS]

    seed_indices = interacted_indices[
        seed_order
    ]

    seen = set(interacted_indices)

    scores = {}

    for seed_idx in seed_indices:

        seed_position = np.where(
            interacted_indices == seed_idx
        )[0][0]

        seed_weight = interacted_weights[
            seed_position
        ]

        distances, indices = model[
            "knn"
        ].kneighbors(
            model["item_matrix"][seed_idx],
            n_neighbors=N_NEIGHBORS
        )

        for distance, neighbor_idx in zip(
            distances[0],
            indices[0]
        ):

            if neighbor_idx in seen:
                continue

            similarity = 1 - distance

            scores[neighbor_idx] = (
                scores.get(
                    neighbor_idx,
                    0
                )
                + similarity * seed_weight
            )

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    result = [
        model["idx_to_item"][item_idx]
        for item_idx, score in ranked[:k]
    ]

    if len(result) < k:

        for item in model["popularity"]:

            if item not in result:
                result.append(item)

            if len(result) == k:
                break

    return result


df = load_data()

model = build_model(df)


st.title(
    "E-Commerce Recommendation System"
)

st.caption(
    "Customer behavior analysis and item-based collaborative filtering"
)


total_users = df["visitorid"].nunique()

total_items = df["itemid"].nunique()

transactions = (
    df["event"] == "transaction"
).sum()

total_events = len(df)


col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Events",
    f"{total_events:,}"
)

col2.metric(
    "Users",
    f"{total_users:,}"
)

col3.metric(
    "Products",
    f"{total_items:,}"
)

col4.metric(
    "Transactions",
    f"{transactions:,}"
)


st.divider()


st.subheader(
    "Customer Funnel"
)

event_counts = df[
    "event"
].value_counts()

views = event_counts.get(
    "view",
    0
)

carts = event_counts.get(
    "addtocart",
    0
)

purchases = event_counts.get(
    "transaction",
    0
)

funnel = pd.DataFrame({
    "Stage": [
        "View",
        "Add to Cart",
        "Transaction"
    ],
    "Events": [
        views,
        carts,
        purchases
    ]
})

st.bar_chart(
    funnel.set_index("Stage")
)


col1, col2, col3 = st.columns(3)

col1.metric(
    "View → Cart",
    f"{carts / views * 100:.2f}%"
)

col2.metric(
    "Cart → Purchase",
    f"{purchases / carts * 100:.2f}%"
)

col3.metric(
    "View → Purchase",
    f"{purchases / views * 100:.2f}%"
)


st.divider()


st.subheader(
    "Model Evaluation"
)

results = pd.DataFrame({
    "Model": [
        "Popularity Baseline",
        "Item-Based Recommendation"
    ],
    "Precision@10": [
        0.0002,
        0.0064
    ],
    "Recall@10": [
        0.0012,
        0.0348
    ],
    "HitRate@10": [
        0.0020,
        0.0615
    ]
})

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True
)


st.divider()


st.subheader(
    "Product Recommendations"
)

default_user = int(
    next(
        iter(
            model["user_to_idx"]
        )
    )
)

user_id = st.number_input(
    "Visitor ID",
    min_value=0,
    value=default_user,
    step=1
)

if st.button(
    "Generate Recommendations"
):

    recommendations = recommend(
        int(user_id),
        model,
        10
    )

    history = (
        model["filtered"][
            model["filtered"]["visitorid"]
            == int(user_id)
        ]
        .sort_values(
            "timestamp",
            ascending=False
        )
        [
            [
                "itemid",
                "event",
                "datetime"
            ]
        ]
        .head(10)
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            "#### Recent Activity"
        )

        if len(history) > 0:

            st.dataframe(
                history,
                use_container_width=True,
                hide_index=True
            )

        else:

            st.info(
                "No filtered interaction history found."
            )

    with col2:

        st.markdown(
            "#### Top 10 Recommendations"
        )

        recommendation_df = pd.DataFrame({
            "Rank": range(
                1,
                len(recommendations) + 1
            ),
            "Item ID": recommendations
        })

        st.dataframe(
            recommendation_df,
            use_container_width=True,
            hide_index=True
        )


st.divider()


st.subheader(
    "Most Active Products"
)

top_items = (
    df.groupby("itemid")
    .size()
    .sort_values(
        ascending=False
    )
    .head(10)
    .reset_index(
        name="Interactions"
    )
)

st.dataframe(
    top_items,
    use_container_width=True,
    hide_index=True
)