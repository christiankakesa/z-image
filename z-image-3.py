import torch
import os
import sys
from datetime import datetime
from diffusers import DiffusionPipeline, PipelineQuantizationConfig
from compel import CompelForSDXL
import random

# 0. RTX 5070 Speed Boosts
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    print("🚨 ERROR: CUDA is NOT available!")
    sys.exit(1)

# 1. Setup
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

static_fallback = "A majestic battle arena for a roguelike action survivor game with a snowy biome."
prompt = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else static_fallback

device, dtype = "cuda", torch.bfloat16
print(f"🔥 Firing up the big guns: {torch.cuda.get_device_name(0)}")

# ---> THE BITSANDBYTES MAGIC <---
# Base Model Config (Compresses UNet and both Text Encoders, leaves VAE untouched)
base_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": dtype
    },
    components_to_quantize=["unet", "text_encoder", "text_encoder_2"]
)

# Refiner Model Config (Refiner only has one text encoder)
refiner_quant_config = PipelineQuantizationConfig(
    quant_backend="bitsandbytes_4bit",
    quant_kwargs={
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": dtype
    },
    components_to_quantize=["unet", "text_encoder_2"]
)

print("🗜️  Quantizing the Transformer down to ~3GB...")
base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
refiner_model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

base = DiffusionPipeline.from_pretrained(
    base_model_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True, quantization_config=base_quant_config
)

if hasattr(base, "vae"):
    base.vae.enable_tiling()
    base.vae.to("cuda")

print("🧠 Setting up Compel to bypass the 77-token limit...")
compel_base = CompelForSDXL(base)
print("📜 Encoding long prompt...")
conditioning = compel_base(prompt)

refiner = DiffusionPipeline.from_pretrained(
    refiner_model_id,
    text_encoder_2=base.text_encoder_2,
    vae=base.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
    quantization_config=refiner_quant_config
)

# 4. Generate
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_path = os.path.join(output_dir, f"z-image-stable-diffusion-xl-{timestamp}.png")

print("🚀 Generating image at lightspeed...")
seed = random.randint(0, 4294967295)
generator = torch.Generator(device).manual_seed(seed)

n_steps = 20
high_noise_frac = 0.8

with torch.inference_mode():
    image = base(
        prompt_embeds=conditioning.embeds,
        pooled_prompt_embeds=conditioning.pooled_embeds,
        num_inference_steps=n_steps,
        denoising_end=high_noise_frac,
        output_type="latent",
        generator=generator,
    ).images
    short_prompt = " ".join(prompt.split()[:50])
    image = refiner(
        prompt=short_prompt,
        num_inference_steps=n_steps,
        denoising_start=high_noise_frac,
        image=image,
        generator=generator,
    ).images[0]

image.save(save_path)
print(f"✅ Image saved to: {save_path}")
