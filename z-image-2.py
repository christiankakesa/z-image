import torch
import os
import sys
from datetime import datetime
from diffusers import ZImagePipeline, ZImageTransformer2DModel, BitsAndBytesConfig

# 0. RTX 5070 Speed Boosts
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

if not torch.cuda.is_available():
    print("🚨 ERROR: CUDA is NOT available!")
    sys.exit(1)

# 1. Setup
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

static_fallback = "A majestic battle arena for a roguelike action survivor game with a lava biome."
prompt = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else static_fallback

device, dtype = "cuda", torch.bfloat16
print(f"🔥 Firing up the big guns: {torch.cuda.get_device_name(0)}")

# ---> THE BITSANDBYTES MAGIC <---
# We configure the 12GB model to be mathematically compressed into 4-bit space (NF4)
# while still doing calculations in bfloat16 to maintain image quality.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=dtype
)

print("🗜️  Quantizing the Transformer down to ~3GB...")
model_name_id="Tongyi-MAI/Z-Image-Turbo"
# Load ONLY the heavy transformer block and compress it
transformer = ZImageTransformer2DModel.from_pretrained(
    model_name_id,
    subfolder="transformer",
    quantization_config=quant_config,
    torch_dtype=dtype,
    device_map=device,
)

# 2. Load the Pipeline
# We pass our newly shrunken transformer into the main pipeline
pipe = ZImagePipeline.from_pretrained(
    model_name_id,
    transformer=transformer, 
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
    device_map=device,
)

# 3. Memory Management
pipe.enable_model_cpu_offload()
pipe.transformer.set_attention_backend("native")

if hasattr(pipe, "vae"):
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

# 4. Generate
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_path = os.path.join(output_dir, f"z-image-turbo-{timestamp}.png")

print("🚀 Generating image at lightspeed...")
seed = torch.seed()
generator = torch.Generator(device).manual_seed(seed)
with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=9,
        guidance_scale=0.0,
        max_sequence_length=1024,
        generator=generator,
    ).images[0]

image.save(save_path)
print(f"✅ Image saved to: {save_path}")
