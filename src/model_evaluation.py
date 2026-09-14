import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors


DATA_PATH = "data/raw/events.csv"

MIN_USER_EVENTS = 7
MIN_ITEM_EVENTS = 15

K = 10
N_NEIGHBORS = 25
SEED_ITEMS = 5
EVAL_USERS = 2000

EVENT_WEIGHTS = {
    "view": 1,
    "addtocart": 3,
    "transaction": 5
}


print("Loading data...")

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

print("Raw rows:", len(df))

df = df.drop_duplicates()

df["weight"] = df["event"].map(EVENT_WEIGHTS)

print("Filtering inactive users...")

user_counts = df["visitorid"].value_counts()

active_users = user_counts[
    user_counts >= MIN_USER_EVENTS
].index

df = df[
    df["visitorid"].isin(active_users)
].copy()

print("Filtering low-interaction items...")

item_counts = df["itemid"].value_counts()

active_items = item_counts[
    item_counts >= MIN_ITEM_EVENTS
].index

df = df[
    df["itemid"].isin(active_items)
].copy()

print("Filtered rows:", len(df))
print("Users:", df["visitorid"].nunique())
print("Items:", df["itemid"].nunique())


print("\nCreating temporal train/test split...")

df = df.sort_values(
    ["visitorid", "timestamp"]
)

test = (
    df.groupby("visitorid", group_keys=False)
    .tail(2)
)

train = df.drop(test.index)

print("Train rows:", len(train))
print("Test rows:", len(test))


user_ids = train["visitorid"].unique()
item_ids = train["itemid"].unique()

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


train_matrix = (
    train.groupby(
        ["visitorid", "itemid"],
        as_index=False
    )["weight"]
    .sum()
)

rows = train_matrix["visitorid"].map(user_to_idx)
cols = train_matrix["itemid"].map(item_to_idx)
values = train_matrix["weight"]

user_item_matrix = csr_matrix(
    (
        values,
        (rows, cols)
    ),
    shape=(
        len(user_ids),
        len(item_ids)
    )
)

print(
    "User-item matrix:",
    user_item_matrix.shape
)


print("\nTraining item similarity model...")

item_user_matrix = user_item_matrix.T

knn = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_neighbors=N_NEIGHBORS,
    n_jobs=-1
)

knn.fit(item_user_matrix)

print("Model ready.")


popularity = (
    train.groupby("itemid")["weight"]
    .sum()
    .sort_values(ascending=False)
)

popular_items = popularity.index.tolist()


def popularity_recommend(user_id, k=10):

    seen = set(
        train.loc[
            train["visitorid"] == user_id,
            "itemid"
        ]
    )

    recommendations = []

    for item in popular_items:

        if item not in seen:
            recommendations.append(item)

        if len(recommendations) == k:
            break

    return recommendations


neighbor_cache = {}


def get_neighbors(item_idx):

    if item_idx in neighbor_cache:
        return neighbor_cache[item_idx]

    distances, indices = knn.kneighbors(
        item_user_matrix[item_idx],
        n_neighbors=min(
            N_NEIGHBORS,
            item_user_matrix.shape[0]
        )
    )

    result = []

    for distance, neighbor_idx in zip(
        distances[0],
        indices[0]
    ):

        if neighbor_idx == item_idx:
            continue

        similarity = 1 - distance

        result.append(
            (
                neighbor_idx,
                similarity
            )
        )

    neighbor_cache[item_idx] = result

    return result


def personalized_recommend(user_id, k=10):

    if user_id not in user_to_idx:
        return []

    user_idx = user_to_idx[user_id]

    user_vector = user_item_matrix.getrow(
        user_idx
    )

    interacted_indices = user_vector.indices
    interacted_weights = user_vector.data

    if len(interacted_indices) == 0:
        return []

    seed_order = np.argsort(
        interacted_weights
    )[::-1][:SEED_ITEMS]

    seed_indices = interacted_indices[
        seed_order
    ]

    seen_indices = set(
        interacted_indices
    )

    scores = {}

    for seed_idx in seed_indices:

        seed_position = np.where(
            interacted_indices == seed_idx
        )[0][0]

        seed_weight = interacted_weights[
           seed_position

        ]
        for neighbor_idx, similarity in get_neighbors(
            seed_idx
        ):

            if neighbor_idx in seen_indices:
                continue

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

    recommendations = [
        idx_to_item[item_idx]
        for item_idx, score in ranked[:k]
    ]

    return recommendations


def precision_at_k(
    recommended,
    relevant,
    k=10
):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0

    hits = len(
        set(recommended)
        & set(relevant)
    )

    return hits / k


def recall_at_k(
    recommended,
    relevant,
    k=10
):

    relevant = set(relevant)

    if len(relevant) == 0:
        return 0

    hits = len(
        set(recommended[:k])
        & relevant
    )

    return hits / len(relevant)


def hit_rate_at_k(
    recommended,
    relevant,
    k=10
):

    hit = (
        set(recommended[:k])
        & set(relevant)
    )

    return 1 if hit else 0


test = test[
    test["visitorid"].isin(
        user_to_idx
    )
]

test = test[
    test["itemid"].isin(
        item_to_idx
    )
]


eligible_users = test[
    "visitorid"
].unique()

np.random.seed(42)

if len(eligible_users) > EVAL_USERS:

    evaluation_users = np.random.choice(
        eligible_users,
        EVAL_USERS,
        replace=False
    )

else:

    evaluation_users = eligible_users


print(
    "\nEvaluation users:",
    len(evaluation_users)
)


pop_precision = []
pop_recall = []
pop_hit = []

model_precision = []
model_recall = []
model_hit = []


print("\nEvaluating models...")


for number, user_id in enumerate(
    evaluation_users,
    start=1
):

    relevant = (
        test.loc[
            test["visitorid"] == user_id,
            "itemid"
        ]
        .unique()
        .tolist()
    )

    pop_rec = popularity_recommend(
        user_id,
        K
    )

    model_rec = personalized_recommend(
        user_id,
        K
    )

    pop_precision.append(
        precision_at_k(
            pop_rec,
            relevant,
            K
        )
    )

    pop_recall.append(
        recall_at_k(
            pop_rec,
            relevant,
            K
        )
    )

    pop_hit.append(
        hit_rate_at_k(
            pop_rec,
            relevant,
            K
        )
    )

    model_precision.append(
        precision_at_k(
            model_rec,
            relevant,
            K
        )
    )

    model_recall.append(
        recall_at_k(
            model_rec,
            relevant,
            K
        )
    )

    model_hit.append(
        hit_rate_at_k(
            model_rec,
            relevant,
            K
        )
    )

    if number % 50 == 0:

        print(
            f"{number} users evaluated"
        )


print("\nRESULTS")
print("-" * 40)

print("\nPopularity Baseline")

print(
    "Precision@10:",
    round(
        np.mean(pop_precision),
        4
    )
)

print(
    "Recall@10:",
    round(
        np.mean(pop_recall),
        4
    )
)

print(
    "HitRate@10:",
    round(
        np.mean(pop_hit),
        4
    )
)


print("\nItem-Based Recommendation")

print(
    "Precision@10:",
    round(
        np.mean(model_precision),
        4
    )
)

print(
    "Recall@10:",
    round(
        np.mean(model_recall),
        4
    )
)

print(
    "HitRate@10:",
    round(
        np.mean(model_hit),
        4
    )
)