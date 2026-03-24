import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Enable TF32 for faster math on Ampere+ GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Check CUDA availability
if not torch.cuda.is_available():
    print("🚨 ERROR: CUDA is NOT available!")
    sys.exit(1)

print(f"🔥 Firing up: {torch.cuda.get_device_name(0)}")

model_name = "Qwen/Qwen2.5-7B-Instruct"

# 4-bit quantization config – reduces memory usage to ~4‑5 GB
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# Load model with quantization and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quant_config,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Determine the prompt from command line or fallback
# static_fallback = "Give me a short introduction to large language model."
# static_fallback = (
#     "Aide moi à développer le texte ci-après, pour écrire un article sur l'IA pour LinkedIn et montrer que l'entreprise Blockchain Services Labs fait de la veille pour permettre plus de confidentialité à l'aide à la création numérique :"
#     "---"
#     "Avec un GPU laptop pour travailler avec des modèles AI qui tournent en local ça fonctionne mais on est loin des performances des outils en lignes qui travailles répondent généralement entre 3 à 30 secondes."
#     "Mais globalement ça fonctionne:"
#     "Text-to-Image => 45 secondes en mix RAM + VRAM avec une RTX 5070 Laptop GPU"
#     "Image-to-image => 15 minutes environ en mix RAM + VRAM avec une RTX 5070 Laptop GPU"
#     "Et le screenshot ici qui montre du Text-to-Text avec un modèle Any-to-Any (Qwen) => 42 secondes en full VRAM avec une RTX 5070 Laptop GPU."
#     "Tous ces test avec du Python, il est temps de passer à Go et C++ 😎"
# )
static_fallback = """Améliore ce blog post pour LinkedIn pour le rendre fluide et accessible à tous :

Blockchain Services Labs : Améliorer la Confidentialité dans la Création Numérique avec l'IA.

Dans l'ère de l'intelligence artificielle (IA), la confidentialité et la sécurité sont des préoccupations majeures pour les entreprises et les individus.

À Blockchain Services Labs, nous mettons en place des solutions innovantes qui non seulement protègent les marques et les projets, mais aussi optimisent les performances de la création numérique.

Nous avons récemment exploré les limites de l'IA en local, en utilisant un GPU intégré dans un ordinateur portable (RTX 5075 Laptop GPU - 8GB VRAM).

Bien que cela fonctionne, il est important de noter que les temps de traitement sont encore loin de ceux des outils cloud.

Voici un aperçu des résultats obtenus lors de nos tests (*de meilleures performances peuvent être obtenues avec une RTX 5090 ou équivalentes) :

- Text-to-Image : le processus a pris 45 secondes. Bien que ce ne soit pas optimal, cela reste acceptable.
- Image-to-Image : Pour cette tâche plus complexe, le temps nécessaire s'élève à environ 15 minutes. Cette longueur de temps souligne l'importance de continuer à optimiser les algorithmes et les configurations matérielles pour améliorer les performances. La limitation de 8GB de VRAM nécessite des temps de transfert longs entre la RAM (mémoire principale) et la VRAM (carte graphique).
- Text-to-Text : Avec un modèle de type Any-to-Any, nous avons démontré une performance impressionnante, prenant seulement 42 secondes en pleine VRAM. Cela souligne notre capacité à créer des modèles efficaces et rapides.

Passage à Go et C++ pour Optimiser les Performances au lieu de Python :

Bien que ces résultats soient prometteurs, nous savons qu'il y a encore du travail à faire.

C'est pourquoi nous avons initié une transition vers des langages de programmation comme Go et C++ (et possiblement Rust), connus pour leur rapidité et leur efficacité en termes de performance et de sécurité.

En passant à ces langages, et uniquement pour les traitements hors GPU, nous espérons atteindre des performances encore plus élevées, ce qui devrait profiter directement à nos clients et à l'industrie dans son ensemble.

À Blockchain Services Labs, nous croyons fermement que la veille technologique est une clé essentielle pour rester en avance sur les tendances et les défis de l'IA.

En continuant à explorer les limites de la technologie et en cherchant constamment des moyens d'améliorer la sécurité et la confidentialité, nous visons à fournir des solutions innovantes qui répondent aux besoins de nos clients dans un monde de plus en plus connecté et automatisé."""
prompt = sys.argv[1].strip() if len(sys.argv) > 1 and sys.argv[1].strip() else static_fallback

# Build chat messages (system + user)
messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize and move to model device
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# Generate response
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096
)

# Trim input tokens from the generated output
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

# Decode the response
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

# Beautify output
print("\n" + "=" * 60)
print("📝 USER PROMPT:")
print("=" * 60)
print(prompt)
print("\n" + "=" * 60)
print("🤖 MODEL RESPONSE:")
print("=" * 60)
print(response)
print("=" * 60 + "\n")