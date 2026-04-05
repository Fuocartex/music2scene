
import librosa
import soundfile as sf
import time
from pathlib import Path



def stream_and_slice(audio_path, out_dir, slice_sec=2.0, sr=16000):
    
    Path(out_dir).mkdir(exist_ok=True)

    # carica audio
    y, _ = librosa.load(audio_path, sr=sr, mono=True)

    total_duration = len(y) / sr
    print(f"Durata audio: {total_duration:.2f}s")

    slice_samples = int(slice_sec * sr)
    idx = 0
    slice_id = 0

    start_time = time.time()

    

    while idx + slice_samples <= len(y):

        # aspetta il tempo "reale"
        elapsed = time.time() - start_time
        expected = slice_id * slice_sec

        if elapsed < expected:
            time.sleep(expected - elapsed)

        # estrai slice
        chunk = y[idx:idx + slice_samples]

        out_path = Path(out_dir) / f"slice_{slice_id:05d}.wav"
        sf.write(out_path, chunk, sr)

        print(f"Salvato: {out_path}")

        idx += slice_samples
        slice_id += 1
        


if __name__ == "__main__":
    stream_and_slice(
        audio_path="input.wav",
        out_dir="live_slices",
        slice_sec=8.0
    )