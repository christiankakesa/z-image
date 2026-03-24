import os
import sys
from datetime import datetime
from PIL import Image
import torch
from diffusers import QwenImageEditPipeline

# -------------------------
# ⚡ GLOBAL OPTIMIZATIONS
# -------------------------
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

if not torch.cuda.is_available():
    raise RuntimeError("🚨 CUDA is NOT available!")

device = torch.device("cuda")
dtype = torch.bfloat16

print(f"🔥 Using GPU: {torch.cuda.get_device_name(0)}")

# -------------------------
# ⚡ LOAD PIPELINE (FASTER)
# -------------------------
print("🗜️ Loading pipeline...")
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit",
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
)

# pipeline.enable_model_cpu_offload()
pipeline.enable_sequential_cpu_offload()

# Memory optimizations
pipeline.vae.enable_tiling()
pipeline.vae.enable_slicing()

# Better than attention slicing on modern GPUs
try:
    pipeline.enable_xformers_memory_efficient_attention()
except Exception:
    # pipeline.enable_attention_slicing()
    pipeline.enable_attention_slicing("max")

pipeline.set_progress_bar_config(disable=True)

print("✅ Pipeline ready")

# -------------------------
# 📂 I/O SETUP
# -------------------------
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

image_path = "./iwin.jpeg"
if not os.path.exists(image_path):
    raise FileNotFoundError(f"🚨 Missing image: {image_path}")

# -------------------------
# 🧠 PROMPT HANDLING
# -------------------------
fallback_prompt = (
    "Transform the subject into a Sasak princess from Lombok, Indonesia. "
    "Traditional Baju Lambung with gold Songket embroidery, floral Lys crown, "
    "ultra detailed textile, cinematic lighting, 8k"
)

prompt = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else fallback_prompt

# -------------------------
# 🖼️ IMAGE PREPROCESS (FASTER)
# -------------------------
image = Image.open(image_path).convert("RGB") #.resize((1024, 1024), Image.LANCZOS)

# -------------------------
# 🎲 SEED (OPTIONAL FIXED FOR REPRO)
# -------------------------
seed = torch.seed()  # faster than random + manual_seed combo
generator = torch.Generator(device=device).manual_seed(seed)

# -------------------------
# ⚡ INFERENCE
# -------------------------
print("🚀 Generating...")

inputs = dict(
    image=image,
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted face, modern clothes, western dress, messy, artifacts, noise, deformed",
    # negative_prompt="",
    true_cfg_scale=4.0,
    num_inference_steps=15,
    generator=generator,
)

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_path = os.path.join(output_dir, f"z-image-qwen-image-edit-{timestamp}.png")

with torch.inference_mode(), torch.amp.autocast('cuda', dtype=dtype):
    result = pipeline(**inputs)
    result.images[0].save(save_path)

print(f"✅ Saved: {os.path.abspath(save_path)}")