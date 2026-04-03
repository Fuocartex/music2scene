import torch
import numpy as np
import pandas as pd
from pathlib import Path
from adapter import AudioToPromptAdapter
from sklearn.metrics.pairwise import cosine_similarity

# Imposta qui i percorsi
CSV = "data/musiccaps/with_embeddings.csv"
EMB = "data/musiccaps/clap_embeddings.npy"
CKPT = "checkpoints_final/adapter_best.pt"

def cosine_mat(matrix):
    cs = cosine_similarity(matrix)
    triu = cs[np.triu_indices_from(cs, k=1)]
    return triu.mean(), triu.std()

def main():
    print("Diagnostica diversità adapter")
    assert Path(CSV).exists(), f"File mancante: {CSV}"
    assert Path(EMB).exists(), f"File mancante: {EMB}"
    assert Path(CKPT).exists(), f"Checkpoint mancante: {CKPT}"

    df = pd.read_csv(CSV)
    embs = np.load(EMB)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Dataset: {len(embs)} embedding audio")
    print("Carico adapter...")
    adapter = AudioToPromptAdapter(audio_dim=512,
    n_tokens=4,
    hidden_dim=768,
    hidden=512,   # <-- stesso valore del training!
    n_layers=3,    # <-- stesso valore del training!
    dropout=0.1,
    use_residual=True).to(device)
    adapter.load_state_dict(torch.load(CKPT, map_location=device))
    adapter.eval()

    # campione casuale di max 50 clip per non saturare memoria
    idx = np.arange(len(embs))
    #idx = np.random.choice(len(embs), size=min(50, len(embs)), replace=False)
    sample_embs = torch.tensor(embs[idx], dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = adapter(sample_embs).cpu().numpy()

    # riduci prompt_embeds (B, n_tokens, 768) → media su token
    avg_outs = outputs.mean(axis=1)  # (B, 768)

    # Varianza dei CLAP embeddings
    var_clap = np.mean(np.std(embs, axis=0))
    print(f"\nVarianza CLAP: {var_clap:.4f}")
    if var_clap < 0.02:
        print("CLAP embeddings poco vari -> modello audio troppo piatto.")

    # Varianza output adapter
    var_adapter = np.mean(np.std(avg_outs, axis=0))
    print(f"Varianza adapter output: {var_adapter:.4f}")
    if var_adapter < 0.01:
        print("Adapter produce embedding quasi identici tra clip diversi.")

    # Similarità media tra audio diversi (cosine)
    mean_cos, std_cos = cosine_mat(avg_outs)
    print(f"Similarità media tra embedding adapter: {mean_cos:.3f} ± {std_cos:.3f}")
    if mean_cos > 0.95:
        print("Tutti gli embedding molto simili → bassa diversità.")
    elif mean_cos > 0.8:
        print("Alcune differenze, ma ancora limitate.")
    else:
        print("Buona diversità, il modello distingue i suoni.")

    # Confronto audio vs output: quanto l’adapter cambia la distribuzione
    mean_cos_a2o, _ = cosine_mat(np.concatenate([embs[idx], avg_outs], axis=1))
    print(f"\nSimilarità media audio->output (grezza): {mean_cos_a2o:.3f}")

    print("\nInterpretazione rapida:")
    print(" - Se var_clap < 0.02 → CLAP non distingue abbastanza gli audio.")
    print(" - Se var_adapter < 0.01 → l'adapter non impara variazioni semantiche.")
    print(" - Se cosine > 0.95 → embedding quasi identici (immagini simili).")
    print(" - Se tutto ok → problema a valle (Stable Diffusion o seed fisso).")

if __name__ == "__main__":
    main()
