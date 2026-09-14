def precision_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    if not recommended:
        return 0.0
    hits = len(set(recommended) & set(relevant))
    return hits / len(recommended)


def recall_at_k(recommended, relevant, k=10):
    if not relevant:
        return 0.0
    recommended = recommended[:k]
    hits = len(set(recommended) & set(relevant))
    return hits / len(set(relevant))


def hit_rate_at_k(recommended, relevant, k=10):
    recommended = recommended[:k]
    return float(bool(set(recommended) & set(relevant)))
