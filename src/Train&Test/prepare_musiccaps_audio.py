import argparse, os, re
from pathlib import Path
from datasets import load_dataset
import yt_dlp
import librosa, soundfile as sf
import pandas as pd
from tqdm import tqdm

KEYWORDS = [
    "classical","orchestra","symphony","sonata","baroque",
    "romantic","chamber","concerto","string quartet","quartet",
    "string orchestra","piano","violin","cello","harpsichord"
]

def is_classical(ex):
    txt = (ex["caption"] or "").lower() + " " + " ".join(ex["aspect_list"] or []).lower()
    return any(k in txt for k in KEYWORDS)

def download_audio(ytid, raw_dir):
    url = f"https://www.youtube.com/watch?v={ytid}"
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(raw_dir / f"{ytid}.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "noplaylist": True,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "m4a"}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        # trova il file scaricato
        for ext in (".m4a", ".mp3", ".webm"):
            p = raw_dir / f"{ytid}{ext}"
            if p.exists():
                return str(p)
    except Exception as e:
        print(f"[SKIP] {ytid} download error: {e}")
    return None

def slice_to_wav(src_path, dst_path, start_s, end_s, sr=48000):
    try:
        y, _ = librosa.load(src_path, sr=sr, mono=True)
        s = int(start_s * sr); e = int(end_s * sr)
        if e <= s or s >= len(y): return False
        e = min(e, len(y))
        seg = y[s:e]
        Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst_path, seg, sr)
        return True
    except Exception as e:
        print(f"[SKIP] slicing error {src_path}: {e}")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/musiccaps", help="cartella di output")
    ap.add_argument("--max_items", type=int, default=10, help="numero massimo di clip da preparare")
    ap.add_argument("--only_classical", action="store_true", help="filtra per classica")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    raw_dir = out_dir / "raw"
    clips_dir = out_dir / "clips"
    out_csv = out_dir / "musiccaps_classical_local.csv"

    print("⬇️ Carico MusicCaps…")
    ds = load_dataset("google/MusicCaps", split="train")
    #if args.only_classical:
     #   ds = ds.filter(is_classical)
    print("→ campioni disponibili:", len(ds))

    rows = []
    n_done = 0
    for ex in tqdm(ds, desc="Prep"):
        if n_done >= args.max_items: break
        ytid = ex["ytid"]
        start_s = float(ex["start_s"]); end_s = float(ex["end_s"])
        caption = ex["caption"]
        aspects = "|".join(ex["aspect_list"]) if ex["aspect_list"] else ""
        # 1) scarica audio
        a_path = download_audio(ytid, raw_dir)
        if not a_path: 
            continue
        # 2) taglia il segmento
        clip_name = f"{ytid}_{int(start_s)}_{int(end_s)}.wav"
        clip_path = str(clips_dir / clip_name)
        ok = slice_to_wav(a_path, clip_path, start_s, end_s, sr=48000)
        if not ok:
            continue
        rows.append({
            "local_audio_path": clip_path,
            "caption": caption,
            "aspect_list": aspects,
            "ytid": ytid,
            "start_s": start_s,
            "end_s": end_s
        })
        n_done += 1

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"✅ Creato {out_csv} con {len(rows)} righe.")
    print(f"📁 Clip in: {clips_dir}")

if __name__ == "__main__":
    main()

