import torch
from PIL import Image
from transformers import (
    GPT2Tokenizer,
    CLIPProcessor,
    CLIPModel
)
from model import ArtTitleGenerator
from collections import Counter

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = "checkpoints/model_epoch_3.pt"  
PREFIX_LEN = 10
MAX_GEN_LEN = 12


tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

clip_model = CLIPModel.from_pretrained(
    "openai/clip-vit-base-patch32"
).to(DEVICE)
clip_processor = CLIPProcessor.from_pretrained(
    "openai/clip-vit-base-patch32"
)

clip_model.eval()

model = ArtTitleGenerator(
    prefix_len=PREFIX_LEN,
    freeze_gpt=True
).to(DEVICE)

model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

def build_allowed_token_ids(titles, tokenizer, top_k=3000):
    all_tokens = []
    for t in titles:
        all_tokens.extend(tokenizer.tokenize(t))

    most_common = Counter(all_tokens).most_common(top_k)

    allowed_ids = {
        tokenizer.convert_tokens_to_ids(tok)
        for tok, _ in most_common
        if tokenizer.convert_tokens_to_ids(tok) is not None
    }

    return allowed_ids

data = torch.load("art_clip_embeddings.pt")
titles = data["titles"]

allowed_token_ids = torch.tensor(
    list(
        build_allowed_token_ids(
            titles,
            tokenizer,
            top_k=3000
        )
    ),
    dtype=torch.long,
    device=DEVICE
)


def embed_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)

    return emb


@torch.no_grad()
@torch.no_grad()
def generate_title(image_emb):
    prefix = model.prefix(image_emb)  # [1, prefix_len, hidden_dim]
    input_ids = torch.empty((1, 0), dtype=torch.long, device=DEVICE)

    for _ in range(MAX_GEN_LEN):
        if input_ids.size(1) > 0:
            token_embeds = model.gpt.transformer.wte(input_ids)
            inputs_embeds = torch.cat([prefix, token_embeds], dim=1)
        else:
            inputs_embeds = prefix

        outputs = model.gpt(inputs_embeds=inputs_embeds)
        logits = outputs.logits[:, -1, :]   # FIXED

        # Vocabulary mask
        mask = torch.full_like(logits, float("-inf"))
        mask[:, allowed_token_ids] = 0
        logits = logits + mask

        logits = logits / 0.7

        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_probs[cumulative_probs > 0.9] = 0

        total_prob = sorted_probs.sum()

        if total_prob <= 0 or torch.isnan(total_prob):
            next_token = sorted_indices[:, 0:1]
        else:
            sorted_probs = sorted_probs / total_prob
            sampled = torch.multinomial(sorted_probs, 1)
            next_token = sorted_indices.gather(-1, sampled)

        input_ids = torch.cat([input_ids, next_token], dim=1)

        if input_ids.size(1) >= 10:
            break

    return tokenizer.decode(
        input_ids.squeeze(),
        skip_special_tokens=True
    )


if __name__ == "__main__":
    import sys
    image_path = sys.argv[1]

    image_emb = embed_image(image_path)
    title = generate_title(image_emb)

    print("\nGenerated title:")
    print(title)
