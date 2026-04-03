# src/precompute_clap_embeddings.py
import argparse
import os
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


from clap_wrapper import ClapWrapper

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="data/musiccaps/musiccaps_classical_local.csv")
    p.add_argument("--out_npy", default="data/musiccaps/clap_embeddings.npy")
    p.add_argument("--out_csv", default="data/musiccaps/with_embeddings.csv")
    p.add_argument("--sr", type=int, default=48000)
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    cw = ClapWrapper()

    embeddings = []
    rows = []
    for i, row in enumerate(tqdm(df.itertuples(index=False), total=len(df))):
        audio_path = getattr(row, "local_audio_path", None) or getattr(row, "audio_path", None)
        if not isinstance(audio_path, str) or not Path(audio_path).exists():
            print(f"[SKIP] missing file {audio_path} (row {i})")
            embeddings.append(np.zeros((512,), dtype=np.float32))
            rows.append(dict(row._asdict(), embedding_index=i))
            continue
        try:
            emb = cw.embed_audio(audio_path, sr=args.sr)  # returns numpy (512,)
            embeddings.append(np.asarray(emb, dtype=np.float32))
            rows.append(dict(row._asdict(), embedding_index=i))
        except Exception as e:
            print(f"[ERROR] embedding failed for {audio_path}: {e}")
            embeddings.append(np.zeros((512,), dtype=np.float32))
            rows.append(dict(row._asdict(), embedding_index=i))

    embeddings = np.stack(embeddings, axis=0)  # (N, 512)
    np.save(args.out_npy, embeddings)
    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print("Saved:", args.out_npy, args.out_csv, "shape:", embeddings.shape)

if __name__ == "__main__":
    main()
