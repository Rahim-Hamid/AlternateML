import pandas as pd
from pathlib import Path
import re
from PIL import Image
import matplotlib.pyplot as plt

# Load CSV
df = pd.read_csv("classes.csv")

df = df[["filename", "artist", "genre", "description"]]
df = df.dropna(subset=["filename", "description"])

# Image root
img_root = Path("E:/wikiart_images")
df["image_path"] = df["filename"].apply(lambda x: img_root / x)

assert df["image_path"].iloc[0].exists()

# Parse title
def parse_title(desc):
    title = desc.replace("-", " ")
    title = re.sub(r"\b(18|19|20)\d{2}\b", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title.title()

df["title"] = df["description"].apply(parse_title)

# Title type
def title_type(title):
    t = title.lower()
    if t.startswith("untitled"):
        return "untitled"
    if t.isdigit():
        return "date"
    return "descriptive"

df["title_type"] = df["title"].apply(title_type)

# Training subset
train_df = df[
    (df["title_type"] == "descriptive") &
    (df["title"].str.len() >= 5)
].sample(n=min(8000, len(df)), random_state=42)

train_df.to_csv("art_titles_clean.csv", index=False)

# Visual check
img = Image.open(train_df.iloc[0]["image_path"]).convert("RGB")
plt.imshow(img)
plt.axis("off")