
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import matplotlib.pyplot as plt


def analyze_embeddings(npy_path):
    npy_path = Path(npy_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"❌ File non trovato: {npy_path}")

    print(f"Carico embeddings da: {npy_path}")
    embs = np.load(npy_path)

    print(f"Shape embeddings: {embs.shape} (N, dim)")

    # -------------------------------
    #  Varianza
    # -------------------------------
    var = np.mean(np.std(embs, axis=0))
    print(f"\n🧩 Varianza media CLAP: {var:.4f}")

    # -------------------------------
    # Similarità cosine
    # -------------------------------
    print("\nCalcolo similarità cosine...")

    cs = cosine_similarity(embs)

    # prendi solo triangolo superiore (senza diagonale)
    vals = cs[np.triu_indices_from(cs, k=1)]

    mean_cos = vals.mean()
    std_cos = vals.std()

    print(f"Cosine similarity media: {mean_cos:.4f} ± {std_cos:.4f}")
    print(f"  min={vals.min():.4f}, median={np.median(vals):.4f}, max={vals.max():.4f}")

    # -------------------------------
    #  Interpretazione automatica
    # -------------------------------
    print("\nInterpretazione:")

    if var < 0.02:
        print("Varianza bassa → audio poco diversi")
    elif var < 0.05:
        print("Varianza moderata")
    else:
        print("Buona varietà negli audio")

    if mean_cos > 0.95:
        print("Embedding quasi identici → dataset problematico")
    elif mean_cos > 0.85:
        print("Audio abbastanza simili")
    else:
        print("Audio ben distribuiti")

    # -------------------------------
    # Istogramma similarità (utile per report)
    # -------------------------------
    print("\nGenero istogramma similarità...")

    plt.figure()
    plt.hist(vals, bins=50)
    plt.title("Distribuzione cosine similarity tra audio")
    plt.xlabel("Cosine similarity")
    plt.ylabel("Frequenza")

    out_plot = npy_path.with_suffix(".png")
    plt.savefig(out_plot)
    print(f"Salvato plot in: {out_plot}")

    print("\nAnalisi completata!")


if __name__ == "__main__":
    # qui analizziamo il dataset
    analyze_embeddings("data/musiccaps/clap_embeddings.npy")