# 🎧 Audio-to-Image Generator

This project generates images from audio using a pre-trained model.


📂 Struttura del progetto

```bash
src/
├── inf1.py
├── adapter.py
├── clap_wrapper.py
├── live_slicing.py
├── live_show.py
├── run.py
│
├── train&test/              
│   ├── train_adapter_robust.py
│   ├── preprocess.py
│   ├── extract_embeddings.py
│   ├── diagnostic_adapter_diversity.py
│   ├── analyze_audio_diversity.py
│   ├── precompute_clap_embeddings.py
│   ├── prepare_musiccaps_audio.py
│

adapter_new.pt
input.wav
requirements.txt
README.md
data/
├── cache/
├── musicacaps/  Contains musicacps row audio and trainng data
├── panns/
├── raw audio/ contains test audio
├── slices/ coontains slice for test of raw audio
```

---

# 📥 1. Clone the repository

```bash
git clone --no-checkout https://github.com/your-repo/audio-to-image.git
cd audio-to-image

git sparse-checkout init --cone
git sparse-checkout set src adapter_new.pt requirements.txt input.wav

git checkout```

---

# ⚙️ 2. Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

---

# 📦 3. Install requirements

```bash
pip install -r requirements.txt
```

---

# ▶️ 4. Run the project (MAIN)

Make sure you have the required files:

```
adapter_new.pt
input.wav
```

Then run:

```bash
python core/run.py
```

This will:

* slice the audio
* generate images
* display them live

---

# 📂 Output

```
live_slices/   → audio chunks
live_frames/   → generated images
```

---

# ⚠️ Notes

* First run will download models (CLAP, Stable Diffusion, CLIP)
* GPU is recommended but not required

---

# 🧪 SECOND PART — Other scripts (optional)

The `train&test/` folder contains additional scripts for training and data processing.

These are NOT required to run the main system. But it required GPU and CUDA.

These were RUN in windows system with CUDA. 

WARNING!! To RUN copy this files in the main folder src. 

---

## Preprocessing audio

```bash
python extra/preprocess.py
```

---

## Extract CLAP embeddings

```bash
python extra/extract_embeddings.py
```

---

## Analyze dataset

```bash
python extra/analyze_audio_diversity.py
```

---

## Train adapter (optional)

```bash
python extra/train_adapter_robust.py
```

---

## Diagnostics

```bash
python extra/diagnostic_adapter_diversity.py
```

---

# 🎯 Summary

* Clone repo
* Create venv
* Install requirements
* Run `src/run.py`

Everything else is optional.

---
