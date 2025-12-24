import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
from pathlib import Path
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

df = pd.read_csv("art_titles_clean.csv")

embeddings = []
titles = []
missing = 0

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = Path(row["image_path"])

    if not image_path.exists():
        missing += 1
        continue

    image = Image.open(image_path).convert("RGB")
    image = Image.open(row["image_path"]).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        emb = model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    embeddings.append(emb.cpu())
    titles.append(row["title"])

torch.save(
    {
        "embeddings": torch.cat(embeddings),
        "titles": titles
    },
    "art_clip_embeddings.pt"
)

print(f"Skipped {missing} missing images")

