import pandas as pd
import re
from bs4 import BeautifulSoup
import warnings
from bs4 import MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

def load_dataset(path):

    df = pd.read_csv(path)

    return df


def rename_columns(df):

    df = df.rename(columns={
        "ProductId": "product_id",
        "Score": "rating",
        "Time": "timestamp",
        "Text": "review_text",
        "HelpfulnessNumerator": "helpful_votes"
    })

    return df


def clean_text(text):

    text = BeautifulSoup(str(text), "html.parser").get_text()

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = text.lower()

    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_reviews(df):

    df = df.dropna(subset=["review_text", "product_id"])

    df["clean_text"] = df["review_text"].apply(clean_text)

    df["word_count"] = df["clean_text"].apply(lambda x: len(x.split()))

    df = df[df["word_count"] >= 20]

    df = df.reset_index(drop=True)

    return df


def convert_timestamp(df):

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    return df


def add_category(df):

    df["category"] = "food"

    return df


def main():

    df = load_dataset("data/amazon_reviews.csv")

    df = rename_columns(df)

    df = preprocess_reviews(df)

    df = convert_timestamp(df)

    df = add_category(df)

    df = df.sample(200000, random_state=42)

    df.to_csv("data/cleaned_amazon_reviews.csv", index=False)

    print("Cleaned dataset saved!")


if __name__ == "__main__":

    main()