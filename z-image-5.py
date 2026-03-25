import sys
import torch
from diffusers import QwenImageEditPipeline
from diffusers.utils import load_image
from nunchaku import NunchakuQwenImageTransformer2DModel
from nunchaku.utils import get_gpu_memory, get_precision

# ====================== CONFIG ======================
rank = 64          # Options: 32 (fastest, lowest VRAM), 64 (good balance), 128 (best quality)
precision = get_precision()   # auto-detects int4 or fp4 for RTX 50-series

# Choose model (recommended for good quality + Lightning support)
model_repo = "nunchaku-tech/nunchaku-qwen-image-edit-2509"
model_name = f"svdq-{precision}_r{rank}-qwen-image-edit-2509.safetensors"

# For 4-step Lightning (super fast ~2-4 seconds):
# model_name = f"svdq-{precision}_r{rank}-qwen-image-edit-lightningv1.0-4steps.safetensors"

print(f"Loading Nunchaku Qwen-Image-Edit | Rank={rank} | Precision={precision}")

# ====================== LOAD QUANTIZED TRANSFORMER ======================
transformer = NunchakuQwenImageTransformer2DModel.from_pretrained(
    f"{model_repo}/{model_name}",
    torch_dtype=torch.bfloat16,
)

# ====================== BUILD PIPELINE ======================
pipeline = QwenImageEditPipeline.from_pretrained(
    "Qwen/Qwen-Image-Edit-2509",   # or "Qwen/Qwen-Image-Edit" for older base
    transformer=transformer,
    torch_dtype=torch.bfloat16,
)

# ====================== 8GB VRAM OFFLOAD (Critical!) ======================
if get_gpu_memory() < 12:   # Your 8GB case
    print("Enabling per-layer offload → ~3-5 GB VRAM expected")
    transformer.set_offload(
        True,
        use_pin_memory=False,
        num_blocks_on_gpu=1          # Try 2 if you have headroom
    )
    if not hasattr(pipeline, "_exclude_from_cpu_offload"):
        pipeline._exclude_from_cpu_offload = []
    pipeline._exclude_from_cpu_offload.append("transformer")
    pipeline.enable_sequential_cpu_offload()   # or enable_model_cpu_offload()
else:
    pipeline.enable_model_cpu_offload()

# VAE optimizations
pipeline.vae.enable_slicing()
pipeline.vae.enable_tiling()

print("✅ Pipeline ready on RTX 5070!")

# ====================== INFERENCE ======================
image = load_image("./iwin.jpeg").convert("RGB")
# image = image.resize((768, 768), Image.Resampling.LANCZOS)  # uncomment if needed

static_fallback = (
    "Transform the subject into a Sasak princess from Lombok, Indonesia. "
    "Traditional Baju Lambung dress with intricate gold Songket embroidery. "
    "Regal crown made of fresh Lys flowers. Highly detailed traditional Indonesian textile patterns. "
    "Hyper-realistic, cinematic lighting."
)
prompt = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else static_fallback
negative_prompt = "blurry, low quality, distorted face, modern clothes, western dress, messy, artifacts, noise"

output = pipeline(
    image=image,
    prompt=prompt,
    negative_prompt=negative_prompt,
    true_cfg_scale=3.5,
    num_inference_steps=4,          # 4 for Lightning, 8-12 otherwise
    generator=torch.Generator("cuda").manual_seed(torch.seed()),
    height=768,
    width=768,
).images[0]

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
save_path = os.path.join(output_dir, f"z-image-nunchaku-{timestamp}.png")
output.save(save_path)
print("✅ Done! Peak VRAM used:", torch.cuda.max_memory_allocated() / 1024**3, "GB")
