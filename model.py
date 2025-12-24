import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel

class ImagePrefix(nn.Module):
    """
    Projects an image embedding into a sequence of prefix token embeddings.
    """
    def __init__(self, image_dim=512, prefix_len=10, hidden_dim=768):
        super().__init__()
        self.prefix_len = prefix_len
        self.hidden_dim = hidden_dim

        self.proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim * prefix_len),
            nn.Tanh()
        )

    def forward(self, image_emb):
        """
        image_emb: [B, image_dim]
        returns:   [B, prefix_len, hidden_dim]
        """
        x = self.proj(image_emb)
        return x.view(-1, self.prefix_len, self.hidden_dim)


class ArtTitleGenerator(nn.Module):
    """
    Multimodal title generator using image-conditioned prefix tuning.
    """
    def __init__(
        self,
        image_dim=512,
        prefix_len=10,   
        gpt_model_name="gpt2",
        freeze_gpt=True
    ):
        super().__init__()

        self.gpt = GPT2LMHeadModel.from_pretrained(gpt_model_name)
        self.hidden_dim = self.gpt.config.n_embd

        for name, param in self.gpt.named_parameters():
            if "h.11" in name or "h.10" in name:
                param.requires_grad = True

        self.prefix = ImagePrefix(
            image_dim=image_dim,
            prefix_len=prefix_len,
            hidden_dim=self.hidden_dim
        )

        self.prefix_len = prefix_len

        if freeze_gpt:
            for param in self.gpt.parameters():
                param.requires_grad = False

    def forward(self, image_emb, input_ids, attention_mask):
        B, T = input_ids.shape

        # 1. Prefix embeddings
        prefix_embeds = self.prefix(image_emb)  

        # 2. Token embeddings
        token_embeds = self.gpt.transformer.wte(input_ids)  

        # 3. Concatenate
        inputs_embeds = torch.cat([prefix_embeds, token_embeds], dim=1)

        # 4. Attention mask
        prefix_mask = torch.ones(
            (B, self.prefix_len),
            device=attention_mask.device
        )
        extended_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        # 5. Labels (ignore prefix)
        prefix_labels = torch.full(
            (B, self.prefix_len),
            -100,
            device=input_ids.device
        )
        labels = torch.cat([prefix_labels, input_ids], dim=1)

        # 6. GPT-2 forward
        outputs = self.gpt(
            inputs_embeds=inputs_embeds,
            attention_mask=extended_mask,
            labels=labels
        )

        return outputs