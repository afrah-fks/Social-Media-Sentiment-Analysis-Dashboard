import pandas as pd
import random

# -----------------------------
# Sample phrases for each class
# -----------------------------

positive_phrases = [
    "I love this product",
    "Amazing experience",
    "Best service ever",
    "Highly recommend this",
    "Absolutely fantastic",
    "Very happy with this",
    "Great quality",
    "Super fast delivery",
    "Worth every penny",
    "Excellent performance"
]

negative_phrases = [
    "Worst experience ever",
    "I hate this product",
    "Very bad service",
    "Not worth the money",
    "Terrible quality",
    "Extremely disappointed",
    "Waste of money",
    "Late delivery",
    "Very खराब experience",  # mix of real-world style
    "Poor customer support"
]

neutral_phrases = [
    "It is okay",
    "Average experience",
    "Nothing special",
    "Product is fine",
    "It works as expected",
    "Delivery was on time",
    "Not bad, not great",
    "Okayish performance",
    "Neutral feeling",
    "Can be improved"
]

# Optional: add random variations
extra_words = ["!", "!!", "...", "really", "very", "kinda", "honestly"]

def add_variation(sentence):
    if random.random() > 0.5:
        return sentence + " " + random.choice(extra_words)
    return sentence

# -----------------------------
# Data Generation Function
# -----------------------------

def generate_data(num_samples=500):
    data = []

    for _ in range(num_samples):
        sentiment = random.choice(["positive", "negative", "neutral"])

        if sentiment == "positive":
            text = random.choice(positive_phrases)
        elif sentiment == "negative":
            text = random.choice(negative_phrases)
        else:
            text = random.choice(neutral_phrases)

        text = add_variation(text)

        data.append([text, sentiment])

    df = pd.DataFrame(data, columns=["text", "sentiment"])
    return df

# -----------------------------
# Generate and Save Dataset
# -----------------------------

if __name__ == "__main__":
    df = generate_data(1000)  # you can change size here
    df.to_csv("data/synthetic_social_media_data.csv", index=False)

    print("✅ Dataset generated successfully!")
    print(df.head())