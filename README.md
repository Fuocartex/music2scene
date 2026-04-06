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
requirements_exact.txt
README.md
```

---

# 📥 1. Clone the repository

```bash
git clone https://github.com/Fuocartex/music2scene.git
```

---

# ⚙️ 2. Create virtual environment

```bash
cd music2scene
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
```

---

# 📦 3. Install requirements

Python version 3.13.12

WARNING!! The requirements file is for running on CPU which is strongly discouraged as the program is very slow at generating images so the real-time part is no longer synchronized.

To run it on GPU, manually install torch with CUDA and the versions of the libraries indicated in requirements_exact (not via file since they are new versions not recognized).

The live show is implemented for windows it may not work on different systems. In the google drive link in the folder output there is a video of how it works. 

```bash
pip install -r requirements.txt
```
Example: To install the new version of torch for CUDA. Check the python version and GPU requirement.
```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```
---

# ▶️ 4. Run the project (MAIN)

Make sure you have the required files:

```
adapter_new.pt
input.wav
```

Then run:
At fisrt run the second cmd could take longer to load the pipeline so let it run. 
```bash
python src/run.py
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

These were RUN on windows system with CUDA. 

To RUN copy this files in the main folder src and download the extra files at these link 
```bash
https://drive.google.com/drive/folders/1_lrUd72yEYG0hUUVuy5NnJK50nRghAti?usp=sharing.


data/
├── cache/
├── musicacaps/  Contains musicacps row audio and trainng data
├── panns/
├── raw audio/ contains test audio
├── slices/ coontains slice for test of raw audio
```
There you can find all previous checkpoint and test, and the Dataset used for training. 

---

## Preprocessing audio

```bash
python src/preprocess.py
```

---

## Extract CLAP embeddings

```bash
python src/extract_embeddings.py
```

---

## Analyze dataset

```bash
python src/analyze_audio_diversity.py
```

---

## Train adapter

```bash
python src/train_adapter_robust.py
```

---

## Diagnostics

```bash
python src/diagnostic_adapter_diversity.py
```

---

# 🎯 Summary

* Clone repo
* Create venv
* Install requirements
* Run `src/run.py`

Everything else is optional.

---
