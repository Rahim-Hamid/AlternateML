import torch

data = torch.load("art_clip_embeddings.pt")
embeddings = data["embeddings"]
titles = data["titles"]

def retrieve_titles(query_emb, k=5):
    query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
    sims = torch.matmul(embeddings, query_emb.T).squeeze()
    topk = sims.topk(k).indices
    return [titles[i] for i in topk]