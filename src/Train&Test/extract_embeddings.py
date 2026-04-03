import argparse
import csv
import numpy as np
from pathlib import Path
from tqdm import tqdm
from clap_wrapper import ClapWrapper
import warnings

def main(args):
    # inizializza CLAP
    cw = ClapWrapper(cache_dir="data/cache")  # usa cache locale se vuoi velocizzare
    rows_out = []
    embeddings = []

    # apertura CSV input
    meta_csv = Path(args.meta_csv)
    if not meta_csv.exists():
        raise FileNotFoundError(f"❌ File CSV non trovato: {meta_csv}")

    with open(meta_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="➡️ Calcolo embedding audio"):
            # prova a leggere il percorso corretto
            audio_path = (
                row.get("local_audio_path")
                or row.get("audio_path")
                or row.get("path")
                or ""
            )
            audio_path = audio_path.strip()
            if not Path(audio_path).exists():
                warnings.warn(f"[SKIP] file mancante: {audio_path}")
                emb = np.zeros((512,), dtype=np.float32)
            else:
                emb = cw.embed_audio(audio_path)  # ora restituisce (512,)

            # aggiungiamo embedding
            embeddings.append(emb)
            row["embedding_index"] = len(embeddings) - 1
            rows_out.append(row)

    # salvataggio numpy
    embeddings = np.stack(embeddings, axis=0)
    Path(args.out_npy).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out_npy, embeddings)
    print(f"✅ Salvati {len(embeddings)} embedding in {args.out_npy} (shape {embeddings.shape})")

    # salvataggio CSV aggiornato
    out_csv = Path(args.out_npy).with_suffix(".csv")
    fieldnames = list(rows_out[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    print(f"✅ CSV aggiornato salvato in {out_csv}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--meta_csv", required=True, help="CSV di input (es. data/musiccaps/musiccaps_classical_local.csv)")
    p.add_argument("--out_npy", default="data/musiccaps/clap_embeddings.npy", help="File .npy di output")
    args = p.parse_args()
    main(args)
