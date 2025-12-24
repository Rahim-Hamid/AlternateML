import torch
from transformers import GPT2Tokenizer
from pathlib import Path

DATA_DIR = Path("data")
EMBED_PATH = "art_clip_embeddings.pt"
OUT_PATH = "art_title_training_data.pt"

MAX_LEN = 12  

data = torch.load(EMBED_PATH)

image_embeddings = data["embeddings"]   
titles = data["titles"]                 

print(f"Loaded {len(titles)} samples")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  

tokenized = tokenizer(
    titles,
    padding="max_length",
    truncation=True,
    max_length=MAX_LEN,
    return_tensors="pt"
)

input_ids = tokenized["input_ids"]         
attention_mask = tokenized["attention_mask"]

assert image_embeddings.size(0) == input_ids.size(0)
assert input_ids.size(1) == MAX_LEN

print("Dataset shapes:")
print("Image embeddings:", image_embeddings.shape)
print("Input IDs:", input_ids.shape)

torch.save(
    {
        "image_embeddings": image_embeddings,
        "input_ids": input_ids,
        "attention_mask": attention_mask
    },
    OUT_PATH
)

print(f"Saved training data to {OUT_PATH}")