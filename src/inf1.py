import torch
from diffusers import StableDiffusionPipeline 
from adapter import AudioToPromptAdapter
from clap_wrapper import ClapWrapper
from transformers import CLIPTokenizer, CLIPTextModel
from PIL import Image
from pathlib import Path
import random
import sys
import select
import time

import msvcrt


class InferencePipeline:
    def __init__(self, adapter_ckpt="adapter_best.pt", device="cuda", n_tokens=16):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Carica adapter

        self.adapter = AudioToPromptAdapter(
            audio_dim=512,
            n_tokens=4, 
            hidden_dim=768,
            hidden=512,
            n_layers=3,
            dropout=0.1,
            use_residual=True
        ).to(self.device)
        self.adapter.load_state_dict(torch.load(adapter_ckpt, map_location=self.device))
        self.adapter.eval()
        
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "prompthero/openjourney-v4",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        self.pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
        self.pipe.enable_attention_slicing()
        
        # 2) Carica il VAE migliore (dopo la pipe!)
        from diffusers import AutoencoderKL

        vae = AutoencoderKL.from_pretrained(
            "stabilityai/sd-vae-ft-mse",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        ).to(self.device)

        # Assegna il VAE alla pipeline
        self.pipe.vae = vae
        # CLAP per audio
        self.clap = ClapWrapper(device=self.device)

        # CLIP per testo base
        
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(self.device).eval()
 
        print(f"Inference running on {self.device} | CUDA: {torch.cuda.is_available()}")
        self.base_prompt = "a single dot"
        

    
    
    def gen_from_audio(self, audio_path, steps=10, guidance=7.5, seed=21, mix_ratio=0.7):  

        if not Path(audio_path).exists():
            raise FileNotFoundError(audio_path)
        
        # Estrai embedding audio
        audio_emb = self.clap.embed_audio(audio_path)
        x = torch.tensor(audio_emb).unsqueeze(0).to(self.device)

        with torch.no_grad():
            audio_prompt_embeds = self.adapter(x)  # (1, n_tokens, 768)

        neg_inputs = self.tokenizer(
            ["photorealistic, realistic, camera, person, face, human, animal, text, letters, watermark, logo, signature, blurry, low quality"], 
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length
        ).to(self.device)


        with torch.no_grad():
            negative_prompt_embeds = self.text_encoder(**neg_inputs).last_hidden_state


        # Prompt testuale fisso → CLIP embedding
        inputs = self.tokenizer([self.base_prompt], return_tensors="pt").to(self.device)
        with torch.no_grad():
            text_emb = self.text_encoder(**inputs).last_hidden_state
 
        final_embeds = torch.cat([audio_prompt_embeds,text_emb], dim=1)

        if negative_prompt_embeds.size(1) != final_embeds.size(1):
            if negative_prompt_embeds.size(1) < final_embeds.size(1):
                pad = torch.zeros(
                    1,
                    final_embeds.size(1) - negative_prompt_embeds.size(1),
                    negative_prompt_embeds.size(2),
                    device=self.device,
                )
                negative_prompt_embeds = torch.cat([negative_prompt_embeds, pad], dim=1)
                print("ops")
            else:
                print("ops2")
                negative_prompt_embeds = negative_prompt_embeds[:, :final_embeds.size(1), :]

        # Genera immagine
        generator = torch.Generator(self.device).manual_seed(seed)
    
        image = self.pipe(
            prompt_embeds=final_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_prompt_embeds.mean(dim=1),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator
        ).images[0]



        return image



if __name__ == "__main__":

    ip = InferencePipeline(adapter_ckpt="adapter_new.pt")

    slices_dir = Path("live_slices")     # input dal processo 1
    out_dir = Path("live_frames")        # output immagini
    out_dir.mkdir(exist_ok=True)

    processed = set()

    seed = 42
    seed = random.randint(0, 2**32 - 1)
    print(seed)
    guidance = 3.5

    print("🚀 Processo 2 avviato (image generation)")

    while True:

        wav_files = sorted(slices_dir.glob("*.wav"))

        for wav_path in wav_files:

            if wav_path in processed:
                continue

            try:
                print(f"🎧 Nuovo slice trovato: {wav_path}")

                img = ip.gen_from_audio(
                    wav_path,
                    steps=10,
                    guidance=3.5,
                    seed=42
                )

                out_name = out_dir / f"{wav_path.stem}.png"
                img.save(out_name)

                print(f" Immagine salvata: {out_name}")

                processed.add(wav_path)

            except Exception as e:
                print(f" Errore con {wav_path}: {e}")

        time.sleep(0.5)  # evita loop aggressivo


"""
3390492916
1868050174 san francesco 
2640770248 san francesco
1902710743 san francesco
4292456150 san francesco
"""