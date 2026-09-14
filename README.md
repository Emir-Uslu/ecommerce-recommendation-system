# E-Commerce Recommendation System

This project analyzes e-commerce user behavior and builds recommendation models from implicit feedback such as product views, cart additions and purchases.

## Stack

Python, pandas, NumPy, scikit-learn, PostgreSQL, SQL, Streamlit

## Project structure

```text
data/
notebooks/
src/
sql/
app.py
```

## Current scope

- Event data cleaning and time-based fields
- User behavior and conversion funnel analysis
- User and item feature generation
- Popularity-based recommendation baseline
- Item-to-item similarity with cosine similarity
- Precision@K, Recall@K and HitRate@K functions
- SQL schema and analytical queries
- Streamlit interface skeleton

## Data

`events_sample.csv` is a small sample dataset used for local development. The final analysis is intended to run on the Retailrocket e-commerce events dataset with the same main event schema.
