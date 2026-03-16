import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json


def load_clean_data():

    df = pd.read_csv("data/cleaned_amazon_reviews.csv")

    return df


def generate_embeddings(texts, batch_size=64):

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):

        batch = texts[i:i+batch_size]

        emb = model.encode(batch, show_progress_bar=False)

        embeddings.append(emb)

    embeddings = np.vstack(embeddings)

    return embeddings


def save_mapping(df):

    mapping = {}

    for i, row in df.iterrows():

        mapping[i] = {
            "product_id": row["product_id"],
            "category": row["category"],
            "rating": row["rating"],
            "review_text": row["clean_text"]
        }

    with open("embeddings/review_id_mapping.json", "w") as f:

        json.dump(mapping, f)


def main():

    df = load_clean_data()

    texts = df["clean_text"].tolist()

    embeddings = generate_embeddings(texts)

    np.save("embeddings/amazon_embeddings.npy", embeddings)

    save_mapping(df)

    print("Embeddings saved!")


if __name__ == "__main__":

    main()