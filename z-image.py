import torch
import os
import sys
from datetime import datetime
from diffusers import ZImagePipeline

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 1. Setup Directory
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

# 2. Input Validation
static_fallback = "A majestic battle arena for a roguelike action survivor game with a snowy biome."
if len(sys.argv) > 1 and sys.argv[1].strip():
    prompt = sys.argv[1].strip()
    print(f"Using provided prompt: {prompt}")
else:
    prompt = static_fallback
    print(f"Using fallback prompt: {prompt}")

# 3. Device & Precision Logic
if torch.cuda.is_available():
    device = "cuda"
    if torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
elif torch.backends.mps.is_available():
    device, dtype = "mps", torch.float16
else:
    device, dtype = "cpu", torch.float32

if device == "cuda":
    print(f"🔥 Firing up the big guns: {torch.cuda.get_device_name(0)}")

# 4. Load Model (Using 'dtype' instead of deprecated 'torch_dtype')
try:
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
except torch.cuda.OutOfMemoryError:
    print("VRAM OOM—try lower resolution or more offloading")

# --- MEMORY SAVING COMMANDS (For 8GB VRAM) ---
if hasattr(pipe, "vae"):
    pipe.vae.enable_tiling() # Prevents the "Killed" crash during final image save
# ------------------------------

# --- MEMORY SAVING COMMANDS ---
if device == "cuda":
    pipe.transformer.set_attention_backend("native")
    torch.backends.cudnn.benchmark = True
    # pipe.enable_model_cpu_offload()
    pipe.enable_sequential_cpu_offload()
else:
    # If on CPU, we must use attention slicing to avoid RAM spikes
    pipe.enable_attention_slicing(1)
# ------------------------------

# 5. File Naming
timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_path = os.path.join(output_dir, f"z-image-{timestamp}.png")

# 6. Generate & Save
print("🚀 Generating image...")
generator = torch.Generator(device).manual_seed(42)
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
