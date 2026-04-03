import os
import argparse
import librosa
import soundfile as sf
from pathlib import Path
import csv
from tqdm import tqdm

def slice_audio(infile, out_dir, window_s, hop_s, sr=16000):
    """Taglia un file audio in finestre di lunghezza window_s con hop hop_s"""
    y, _ = librosa.load(infile, sr=sr, mono=True)
    dur = len(y) / sr
    print("durata:", dur)
    print("window_s:", window_s)
    out_paths = []
    i = 0
    print("check:", i+window_s)
    while i + window_s <= dur:
        start = int(i * sr)
        end = int((i + window_s) * sr)
        seg = y[start:end]
        out_name = Path(out_dir) / f"{Path(infile).stem}_w{int(i):05d}.wav"
        print("out_name:", out_name)
        sf.write(out_name, seg, sr)
        out_paths.append(str(out_name))
        i += hop_s
    return out_paths

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    audio_files = [str(p) for p in Path(args.src_dir).glob("*.*")]
    rows = []
    for a in tqdm(audio_files, desc="Preprocessing audio"):
        out_paths = slice_audio(a, args.out_dir, args.window_s, args.hop_s, args.sr)
        for p in out_paths:
            rows.append([p, "", ""])  # audio_path, image_path, caption
    csv_path = Path(args.out_dir) / "slices_metadata.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["audio_path", "image_path", "caption"])
        writer.writerows(rows)
    print(f"✅ Preprocessing completato, metadata salvato in {csv_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src_dir", required=True, help="Cartella con audio originale")
    p.add_argument("--out_dir", required=True, help="Cartella di output per slices")
    p.add_argument("--window_s", type=float, default=15.0, help="Lunghezza finestra (s)")
    p.add_argument("--hop_s", type=float, default=5.0, help="Hop tra finestre (s)")
    p.add_argument("--sr", type=int, default=16000, help="Sample rate target")
    args = p.parse_args()
    main(args)

