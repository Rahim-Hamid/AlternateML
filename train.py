import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from model import ArtTitleGenerator
from dataset import ArtTitleDataset


DATA_PATH = Path("art_title_training_data.pt")
BATCH_SIZE = 16
LR = 1e-5   
EPOCHS = 3  
PREFIX_LEN = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)


data = torch.load(DATA_PATH)

dataset = ArtTitleDataset(
    image_embeddings=data["image_embeddings"],
    input_ids=data["input_ids"],
    attention_mask=data["attention_mask"]
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Loaded {len(dataset)} training samples")


model = ArtTitleGenerator(
    prefix_len=PREFIX_LEN,
    freeze_gpt=True   
)

model.to(DEVICE)
model.train()

optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

for epoch in range(EPOCHS):
    epoch_loss = 0.0

    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for batch in loop:
        optimizer.zero_grad()

        image_emb = batch["image_emb"].to(DEVICE)
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)

        outputs = model(
            image_emb=image_emb,
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = outputs.loss
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} average loss: {avg_loss:.4f}")

    ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")

print("Training complete.")