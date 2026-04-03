import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel

from adapter import AudioToPromptAdapter


# -------------------------
# DATASET
# -------------------------

class AudioTextDataset(Dataset):
    def __init__(self, csv_path, emb_path, device="cpu"):
        self.df = pd.read_csv(csv_path)
        self.embs = np.load(emb_path)
        self.device = device

        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to(device).eval()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        audio = torch.tensor(self.embs[idx], dtype=torch.float32)

        caption = row.get("caption", "")
        if not isinstance(caption, str):
            caption = ""

        inputs = self.tokenizer(
            [caption],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            text_emb = self.text_encoder(**inputs).last_hidden_state.mean(dim=1)

        return audio, text_emb.squeeze(0).cpu()


# -------------------------
# LOSSES
# -------------------------

def contrastive_loss(a, t, temperature=0.07):
    """
    InfoNCE loss
    """
    a = F.normalize(a, dim=-1)
    t = F.normalize(t, dim=-1)

    logits = a @ t.T / temperature
    labels = torch.arange(a.size(0)).to(a.device)

    loss_a = F.cross_entropy(logits, labels)
    loss_t = F.cross_entropy(logits.T, labels)

    return (loss_a + loss_t) / 2


def diversity_loss(x):
    """
    Penalizza embedding simili nel batch
    """
    if x.size(0) < 2:
        return torch.tensor(0.0, device=x.device)

    x = F.normalize(x, dim=-1)
    sim = x @ x.T

    mask = 1 - torch.eye(sim.size(0), device=x.device)
    return (sim * mask).mean()


def variance_loss(x):
    """
    Evita collapse (varianza troppo bassa)
    """
    std = torch.sqrt(x.var(dim=0) + 1e-6)
    return torch.mean(F.relu(1 - std))


# -------------------------
# TRAIN
# -------------------------

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(" Device:", device)

    dataset = AudioTextDataset(
        "data/musiccaps/with_embeddings.csv",
        "data/musiccaps/clap_embeddings.npy",
        device=device
    )

    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # MODEL MIGLIORATO
    model = AudioToPromptAdapter(
        audio_dim=512,
        n_tokens=4,          
        hidden_dim=768,
        hidden=512,          
        n_layers=3,          
        dropout=0.1,
        use_residual=True
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(40):
        model.train()

        total_loss = 0

        for audio, text in tqdm(loader):
            audio = audio.to(device)
            text = text.to(device)

            pred = model(audio)  # (B,1,768)
            pred = pred.mean(dim=1)    

            # LOSSES
            loss_contrast = contrastive_loss(pred, text)
            loss_div = diversity_loss(pred)
            loss_var = variance_loss(pred)

            loss = loss_contrast + 0.05 * loss_div + 0.05 * loss_var

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch} | Loss: {total_loss/len(loader):.4f}")

        torch.save(model.state_dict(), "adapter_new.pt")


if __name__ == "__main__":
    train()