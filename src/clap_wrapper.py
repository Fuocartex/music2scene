import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModel
from pathlib import Path
import warnings

class ClapWrapper:
    """
    Wrapper robusto per CLAP (Contrastive Language-Audio Pretraining).
    Usa il modello 'laion/clap-htsat-unfused' per estrarre embedding audio.
    """

    def __init__(self, device=None, max_duration=30.0, cache_dir=None):
        """
        Args:
            device: 'cuda' o 'cpu'
            max_duration: durata massima (in secondi) da considerare per clip troppo lunghi
            cache_dir: se specificato, salva e ricarica embedding già calcolati (es. 'data/cache')
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_duration = max_duration
        self.cache_dir = Path(cache_dir) if cache_dir else None

        model_name = "laion/clap-htsat-unfused"
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"CLAP caricato su {self.device} ({model_name})")

    def _load_audio(self, wav_path, sr=48000):
        """Carica e normalizza un file audio in float32 mono."""
        try:
            y, _ = librosa.load(wav_path, sr=sr, mono=True)
        except Exception as e:
            warnings.warn(f"[ERRORE] Caricamento fallito per {wav_path}: {e}")
            return None

        if len(y) == 0:
            warnings.warn(f"[ATTENZIONE] File vuoto: {wav_path}")
            return None

        # Troncamento se troppo lungo
        max_len = int(sr * self.max_duration)
        if len(y) > max_len:
            y = y[:max_len]

        # Normalizzazione ampiezza
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))

        return y.astype(np.float32)

    def embed_audio(self, wav_path, sr=48000, normalize=True, use_cache=True):
        """
        Estrae l'embedding CLAP da un file audio.
        Args:
            wav_path: percorso al file WAV/M4A/MP3
            sr: sample rate target (default 48000)
            normalize: se True, restituisce embedding unitario (norma 1)
            use_cache: se True, usa la cache se disponibile
        Returns:
            np.ndarray di shape (512,)
        """
        wav_path = Path(wav_path)
        if not wav_path.exists():
            warnings.warn(f"[ERRORE] File inesistente: {wav_path}")
            return np.zeros((512,), dtype=np.float32)

        # Cache lookup
        cache_path = None
        if self.cache_dir and use_cache:
            cache_path = self.cache_dir / (wav_path.stem + ".npy")
            if cache_path.exists():
                return np.load(cache_path)

        # Carica audio
        y = self._load_audio(wav_path, sr)
        if y is None:
            return np.zeros((512,), dtype=np.float32)

        # CLAP embedding
        inputs = self.processor(audios=y, sampling_rate=sr, return_tensors="pt", padding=True)
        with torch.no_grad():
            audio_emb = self.model.get_audio_features(
                **{k: v.to(self.device) for k, v in inputs.items()}
            )

        emb = audio_emb.squeeze(0).cpu().numpy().astype(np.float32)

        # Normalizzazione L2
        if normalize:
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb /= norm

        # Cache save
        if cache_path and use_cache:
            np.save(cache_path, emb)

        return emb

    def embed_batch(self, paths, sr=48000, normalize=True):
        """
        Estrae embedding per più file (lista di percorsi).
        Ritorna un array (N, 512)
        """
        embs = []
        for p in paths:
            emb = self.embed_audio(p, sr=sr, normalize=normalize)
            embs.append(emb)
        return np.stack(embs, axis=0)
