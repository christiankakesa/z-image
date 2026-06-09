"""
Local LLM chat script using HuggingFace Transformers + BitsAndBytes 4-bit quantization.
Supports streaming output and multi-turn conversation.
"""

import sys
import re
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)
from typing import Tuple

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

SYSTEM_PROMPT = "You are Qwen, a helpful, concise and professional assistant."

# STATIC_FALLBACK_PROMPT = (
#     "Bonjour, je voudrais que tu m'aides à écrire des histoires fantastiques "
#     "dans un univers Afro-futuriste, inspirées des contes et légendes d'Afrique "
#     "sub-saharienne, avec des rappels de faits historiques réels."
# )
STATIC_FALLBACK_PROMPT = (r"""
\documentclass[16pt,a4paper,titlepage,openright,twoside,final]{book}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{graphicx}
\usepackage{geometry}
\usepackage[french]{babel}
\usepackage[french]{isodate}
\usepackage{xcolor}
\usepackage{fixme}
\usepackage{lmodern}
%% \usepackage{titlesec}

%% \titleformat{\chapter} % titlesec package
%%   {\Large\bfseries} % format
%%   {}                % label
%%   {0pt}             % sep
%%   {\huge}           % before-code
\graphicspath{{images/}} % graphicx package

\title{La Rose d'Afrique}
\author{C.~F.~Kakesa}
\date{}

\begin{document}
\renewcommand{\chaptername}{Chapitre}
\renewcommand{\contentsname}{Table des Matières}
\renewcommand{\bibname}{Bibliographie}
\pagestyle{headings}

\frontmatter
\begin{titlepage}
    \newgeometry{margin=1in} % Center everything perfectly for the cover
    \centering
    
    \vspace*{1cm}
    
    {\Huge \bfseries La Rose d'Afrique \par}
    \vspace*{1cm}
    {\Large \scshape par C.~F.~Kakesa \par}
    
    \vfill
    
    % --- Optional Illustration ---
    % If you have a specific illustration for the center:
    \includegraphics[width=0.6\textwidth]{cover.jpg} 
    
    \vfill

    \vspace*{0.5cm}
    {\large (c) 2015 C.~F.~Kakesa \par}
    
    \restoregeometry % Returns margins to normal for the rest of the book
\end{titlepage}

% Auteurs
% author: Christian Kakesa
% illustrated by: Christian Kakesa
% year: 2015-2026.

\chapter*{Préface}

\vspace*{\fill}
«~Je dédie ce livre à mes 3 filles Rachel, Anaïs et Élisa. ~».
\\
\\
«~Mon inspiration vient de mon enfance vécu en afrique jusqu'à l'âge de 8 ans et de la culture occidentale que j'ai embrassé pour revenir aux sources et comprendre mes racines africaines~».
\vspace*{\fill}

\mainmatter
\chapter{Une famille extraordinaire}
Après une longue journée passée à jouer dans la nature, Neo et Niobe s'apprêtent à rentrer chez eux.
Les deux frères et sœurs aiment jouer ensemble au-delà des plaines et collines de la savane.
Il y règne une atmosphère paisible et lumineuse, remplie d'une énergie protectrice.
\paragraph{}
Niobe, la grande sœur de Neo, veille toujours sur son petit frère.
Neo, quant à lui, veut toujours montrer à sa sœur à quel point il est fort et courageux.
Après cette longue journée, il est temps pour les deux jeunes de rentrer à la maison avant le dîner.
\paragraph{}
Les enfants arrivent devant leur maison.
La porte s'ouvre en présence de nos jeunes aventuriers.
Neo part déposer son arc dans la pièce de rangement des outils, dans une partie excentrée de la maison, accessible depuis l'intérieur.
Niobe, quant à elle, vérifie qu'elle n'a pas perdu son amulette porte-bonheur, que sa grand-mère lui avait donnée et qu'elle garde toujours sur elle.
\paragraph{}
Comme un rituel, ils saluent tout d'abord leur père, qui se réjouit de voir ses enfants en bonne santé après une bonne journée passée dans les plaines d'Afrique centrale.
Ils vont ensuite dans la cuisine pour y rejoindre leur mère, qui les embrasse très chaleureusement.
\paragraph{}
Neo et Niobe commencent par se laver les mains avant de mettre la table.
Une bonne odeur de viande de brousse parfume la maison, ainsi qu'une fine odeur de saka-saka\,\footnote{Légume qui se mange en sauce, à base de feuilles de magnolia. Ce plat se prépare avec de l'huile de palme et s'accompagne de semoule, de foufou, de chikwang, de pain, de bananes plantains ou d'igname\dots}.
Toute la famille passe à table.
C'est un moment privilégié où les parents en profitent pour discuter et transmettre la culture et les coutumes de la famille.

\chapter{Une sacrée soirée}
Ce soir est un grand soir.
Le père de nos deux jeunes a une chose très importante à leur transmettre.
Il s'agit d'une légende dont nulle ne connaît l'origine, mais qui est connue par tous.
Le papa de Niobe et de Neo se prépare à raconter l'histoire de \emph{La Rose d'Afrique}.
\\
\\
--- Mes enfants, il existe une plante qui a le pouvoir immense.
\\
--- Cette plante est utilisée pour sauver les habitants de la planète.
\\
--- Ces vertus magiques sont convoitées par beaucoup de personnes qui n'ont pas toujours de bonnes intentions.
\\
--- Rare, elle ne peut être vue que par une personne qui a du cœur.
\paragraph{}
\emph{La Rose d'Afrique}, ne se trouve qu'en Afrique centrale.
Depuis plusieurs années, seules les membres de leur famille ont été capables de sauver l'humanité en allant trouver \emph{La Rose d'Afrique}.
Les enfants écoutent avec passion leur père raconter cette histoire.
Neo et sa sœur Niobe, aimeraient être les prochains membres de leur famille à ramener \emph{La Rose d'Afrique} pour sauver l'humanité et rendre leur famille fière.

\chapter{Une nouvelle terrifiante}
Le lendemain matin, la maman réveille les enfants pour qu'ils se préparent.
Les enfants ont l'habitude de ranger leurs chambres et de faire quelques exercices de réveil musculaire.
Niobe, qui est très débrouillarde, met la table pour le petit déjeuner.
Neo va chercher son père pour réparer son arc, qui avait subi quelques dégâts légers la veille.
\paragraph{}
Arrivé dans la chambre des parents, le garçon voit son père allongé dans son lit et transpirant à grosses gouttes.
Il s'approche et demande à son père si tout va bien.
Le père de Neo semble mal en point ; il n'arrive plus à parler.
Neo arrive à déchiffrer le message que lui dit son père malgré son état. Parmi les mots que son père prononce, Neo entend : «~Bolasa~».
Le jeune garçon en déduit que son père a la fièvre. Il pense que ce mot pourrait être Bolasa.\,\footnote{Une forte fièvre qu'on ne peut pas guérir (c'est une maladie fictive, une contraction d'Ebola et de Lassa).}.
\paragraph{}
Neo court aussitôt prévenir sa mère et sa sœur de l'état de santé de son père.
Toute la famille est autour de papa.
La maman des deux enfants prend peur et se demande comment un miracle peut sauver son mari.
Soudain, Niobe se souvient de la légende de \emph{La Rose d'Afrique}.
Un simple regard vers son frère suffit pour que les deux enfants se préparent pour aller chercher \emph{La Rose d'Afrique}.
\paragraph{}
Niobe commence par préparer un sac de provisions pour son frère et elle.
Elle espère que le voyage ne sera pas trop long et qu'ils arriveront à temps pour sauver leur père.
Malgré l'état de son arc, Neo se rend à l'armurerie et enfile son arme.
Il est très affecté par l'état de santé de son père, mais garde la tête haute.
Il veut montrer à sa sœur qu'il est fort et qu'il peut faire face à cette épreuve.
Niobe termine les dernières préparations et fait signe à son frère qu'il est temps de partir.

\chapter{Le départ}
La mère, impuissante, regarde ses enfants partir vers une quête dont seule le destin connaît l'issue.
Neo est muni de son arc, tandis que Niobe possède son amulette, qui la rend toujours confiante.
Les enfants sont en route pour trouver \emph{La Rose d'Afrique}.
Ils traversent les plaines chaudes de la savane qui les entoure.
Ils arrivent à l'entrée d'une forêt épaisse, qui ne semble pas faire partie du climat africain.
La température est légèrement plus basse, c'est une particularité de ce lieu presque magique.
\paragraph{}
Les deux enfants pénètrent dans la forêt.
Il fait sombre malgré que le sol soit rempli de taches de lumière.
Un souffle fin va et vient, tantôt soufflant vers l'avant, tantôt vers l'arrière\dots
\paragraph{}
Soudain, l'amulette de Niobe se met à briller d'une lumière blanchâtre étincelante.
C'est la première fois que ce phénomène se produit pour nos jeunes aventuriers.
Neo se demande ce qui arrive à sa sœur, car elle semble plongée dans un rêve.
Avant qu'il puisse finir de se poser des questions sur Niobe, l'arc de Neo s'entoure d'un champ de force verdâtre.
Il se sent également emporté par une force bienveillante. Sans être préparé, Neo peut ressentir l'essence même d'une communion avec la nature.
Après quelques secondes, les deux enfants se reprennent et comprennent qu'ils se rapprochent de leur but.
\paragraph{}
Peu après, une voix retentit au loin.
Les enfants entendent une voix mystérieuse et divine les appeler : 
\\
\\
--- Mes enfants, on me connait sous les noms de Nzambi, d'Olofi, Olorun, et Olodumare\dots
\\
--- Je suis la divinité suprême de tous les peuples\dots
\\
\\
Niobe et Neo sont touchés par la grâce car Olofi n'intervient presque jamais aux humains.
Cette voix leur a redonné du courage pour continuer leur quête.

\chapter{Une lueure d'espoir}
Nos deux aventuriers continuent leur chemin à travers la forêt, guidés par la voix qui les appelle.
À mesure qu'ils avancent, une lumière sous la forme d'un portail spatio-temporel se révèle.
Neo et Niobe se sentent confiants et franchissent le portail.
Ils arrivent dans une clairière où se trouve un magnifique palais.
Le palais est entouré de jardins luxuriants, remplis de fleurs colorées et d'arbres majestueux.
Au centre de la clairière, une fontaine d'eau cristalline jaillit, créant une atmosphère paisible et magique.
Les enfants sont émerveillés par la beauté du lieu, mais ils savent qu'ils doivent se concentrer sur leur quête : \emph{La Rose d'Afrique}.
\paragraph{}
Les enfants s'approchent du palais et sont accueillis par une gardienne à l'allure gerrière.
Elle parle aux enfants une langue étrange, mais grâce à l'amulette de Niobe et la synergie de l'arc de Neo, ils peuvent comprendre ce qu'elle dit.
Les jeunes héros apprennent qu'ils sont attendus depuis très longtemps, tel que la prophétie le prévoit, et que \emph{La Rose d'Afrique} se trouve à l'intérieur du palais.
La guerrière explique qu'elle doit les mener à \emph{la Reine des Guerrières}, une puissante Agojie\,\footnote{Aussi appelées les Amazones du Dahomey, les Agojie sont des guerrières qui ont protégé leurs rois et reines entre le XVIIIe et le XIXe siècle. Elles étaient organisées en factions et strictement composées de femmes.}, qui est la seule à pouvoir leur donner accès à \emph{La Rose d'Afrique}.
Ils traversent le gigantesque palais dont les murs sont ornés de sculptures représentant des scènes de batailles et de cérémonies rituelles.
Ils arrivent à la salle du trône, où se trouve \emph{la Reine des Guerrières}, appelée aussi Reine Tasi.

\chapter{La rencontre des scribes Éthiopiens}
La Reine Tasi, une puissante guerrière au regard doux, accueille les enfants avec respect et leur explique qu'ils doivent passer une épreuve redoutable pour accéder à \emph{La Rose d'Afrique} : l'épreuve de \emph{la Rosa Abisinika}\,\footnote{On l'appelle souvent Rosa abyssinica, ou encore rose d'Éthiopie, rose sauvage d'Afrique… mais elle porte aussi d'autres noms, comme autant de secrets précieux. Cette rose discrète est connue depuis toujours pour ses merveilleuses vertus et ses pouvoirs bienfaisants.}.
L'accès à \emph{La Rose d'Afrique} est protégé par un sort ancestral.
Nos jeunes héros doivent s'en approcher et y être acceptés en premier lieu.
\paragraph{}
Lorsque Neo et Niobe s'approchent de l'autel où l'on peut apercevoir \emph{La Rose d'Afrique}, une lumière à peine visible apparaît assez lentement et disparaît.
La Reine, observant ce phénomène, est de plus en plus rassurée que ces enfants sont bien ceux attendus depuis des années.
\paragraph{}
L'autel où se trouve \emph{La Rose d'Afrique} s'illumine soudainement et nos ados s'en rapprochent naturellement, se voyant ainsi comme emportés dans un rêve.
Neo et Niobe arrivent dans un paysage futuriste avec beaucoup de nature et très coloré qui les entoure.
À leur grande surprise, ils sont accueillis par un groupe d'hommes vêtus tels des prêtres nubiens ; ceux-ci sont les scribes de la Nubie.
Il s'en suit des habitants leur montrant des sourires radieux et un accueil chaleureux dans un calme religieux.

\chapter{L'épreuve de La Rose d'Afrique}
Neo et Niobe sont stupéfaits par la beauté et la grandeur de ce lieu sacré.
Le scribe en chef, un sage, vêtu de robes blanches ornées de symboles anciens, leur récite une prière ancienne avant de leur souhaiter la bienvenue.
Il explique que l'épreuve de \emph{la Rosa Abisinika} consiste à répondre à trois questions qui testent leur courage, leur intelligence et leur compassion.
Les deux adolescents hochent la tête, leur résolution inébranlable peut se lire sur leurs visages.
Le scribe en chef désigne trois autels situés au bout de la grande pièce, chacun éclairé par une lumière dorée.
Chaque autel contient une question gravée dans la pierre, encadrée par des images symboliques.
\paragraph{} 
Les scribes observent attentivement Neo et Niobe et demandent aux enfants de se laisser guider par les lumières dorées émanant des autels.
\\
Le premier autel s'illumine, et une voix douce et profonde résonne : 
\\
--- Combien de fois avez-vous aidé votre frère ou votre sœur sans attendre de retour ? 
\\
Neo, sans hésiter, répond : 
\\
--- Nous nous soutenons mutuellement depuis notre plus jeune âge.
Niobe et moi avons traversé mille aventures ensemble, et chaque fois, nous avons été là l'un pour l'autre. 
\\
Niobe, avec un sourire timide, ajoute : 
\\
--- Notre famille est notre force, et nous sommes prêts à faire face à n'importe quel défi ensemble. 
\\
La voix continue : 
\\
--- Très bien, passez à l'autre autel. 
\\
Le second autel s'éclaire, et une question apparaît : 
\\
--- Quel est le plus grand sacrifice que vous ayez jamais fait pour votre famille ?
\\
Neo réfléchit un instant, puis répond : 
\\
--- Nous offrons de la nourriture et des vivres à ceux qui en ont besoin, suivant l'enseignement de notre mère, et nous le faisons avec bonté.
\\
Niobe ajoute : 
\\
--- Et nous passons notre propre confort personnel pour assurer le bien-être de notre famille.
\\
La voix féminine approuve : 
\\
--- Excellent. Dernier autel, dernier question. 
\\
Le troisième autel s'illumine, et la question apparaît : 
\\
--- Quelle est la plus grande leçon que vous avez apprise de votre famille ?
\\
Neo et Niobe se regardent un instant, puis Neo répond : 
\\
--- Nous avons appris l'importance de la famille et de la communauté.
Nous avons compris que nous ne pouvons pas faire face à tous les défis seuls et que nous devons travailler ensemble pour résoudre les problèmes. 
\\
Niobe ajoute : 
\\
--- Nous avons aussi appris l'importance de la patience, de la persévérance et de la foi en nos capacités. 
\\
La voix féminine s'élève pour conclure : 
\\
--- Vous avez réussi l'épreuve de \emph{la Rosa Abisinika}.
Vous êtes maintenant prêts à recevoir \emph{La Rose d'Afrique}. 
\\
Les deux adolescents échangent un regard d'espérance et de gratitude avant de s'avancer vers l'autel central, où \emph{La Rose d'Afrique} est enfin révélée, radiante de lumière.

\chapter{La famille royale}

\paragraph{}


\chapter{Le sauvetage}

\paragraph{}


\chapter{Fin\dots}
%% Le destin d'une vie


\cleardoublepage{}
%% Résumé de la couverture arrière
Pour sauver leur père malade, Neo et Niobe partent à la conquête de \emph{La Rose d'Afrique}.
Ils traverseront des épreuves difficiles pour y parvenir et vont découvrir les décors magnifiques des vallées africaines.

\end{document}


J'aimerais que tu m'aides à écrire le chapitre 8 de La Rose d'Afrique, ci -dessus, en gardant mon style et la taille des paragraphes, 
le document est en latex et j'aimerais conserver ce format.
Le chapitre 8, La Famille Royale, fait suite à l'obtention de La Rose d'Afrique, où Neo et Niobe doivent s'entraîner dans une dimension parralèle où le temps s'écoule beaucoup moins vite afin d'être en capité d'utiliser les pouvoir de la Rose d'Afrique, découvrir leurs propre pouvoirs et destin, et recevoir la révélation de leur liens avec la famille royale.
"""    )

# Maximum number of turns kept in context (user + assistant pairs).
# Older turns are dropped to avoid exceeding the context window.
MAX_HISTORY_TURNS = 20

# Generation hyperparameters
GENERATION_KWARGS = dict(
    max_new_tokens=4096,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    repetition_penalty=1.1,
)

# ---------------------------------------------------------------------------
# Enable TF32 for faster math on Ampere+ GPUs
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model with 4-bit quantization and its tokenizer."""
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    print(f"⏳ Loading model: {model_name} …")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Some tokenizers lack a pad token — fall back to eos to silence warnings.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("✅ Model loaded.\n")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Response generation (with streaming)
# ---------------------------------------------------------------------------

def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list,
    stream: bool = True,
) -> str:
    """
    Generate a response for the current conversation history.

    Args:
        model: The causal LM.
        tokenizer: Matching tokenizer.
        messages: Full conversation history (list of role/content dicts).
        stream: If True, tokens are printed to stdout as they are generated.

    Returns:
        The assistant's reply as a plain string.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True) if stream else None

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            **GENERATION_KWARGS,
            pad_token_id=tokenizer.pad_token_id,
            streamer=streamer,
        )

    # Decode only the newly generated tokens (exclude the prompt)
    new_ids = [
        out[len(inp):]
        for inp, out in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(new_ids, skip_special_tokens=True)[0]
    return response.strip()


# ---------------------------------------------------------------------------
# Conversation helpers
# ---------------------------------------------------------------------------

def trim_history(messages: list, max_turns: int) -> list:
    """
    Keep the system message and at most `max_turns` user/assistant pairs.
    Older pairs are silently dropped.
    """
    system = [m for m in messages if m["role"] == "system"]
    dialogue = [m for m in messages if m["role"] != "system"]

    # Each "turn" = 1 user msg + 1 assistant msg = 2 items
    max_items = max_turns * 2
    if len(dialogue) > max_items:
        dropped = (len(dialogue) - max_items) // 2
        print(f"⚠️  Context window management: dropping {dropped} oldest turn(s).")
        dialogue = dialogue[-max_items:]

    return system + dialogue


def print_separator(char: str = "─", width: int = 80) -> None:
    print(char * width)


def display_response(response: str, streamed: bool) -> None:
    """Print the response block (skip if already streamed)."""
    print_separator()
    if not streamed:
        print(response)
    print_separator()
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # ── CUDA check ──────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("🚨 ERROR: CUDA is not available. A GPU is required.")
        sys.exit(1)

    print(f"🔥 GPU detected: {torch.cuda.get_device_name(0)}")

    # ── Load model ──────────────────────────────────────────────────────────
    model, tokenizer = load_model(MODEL_NAME)

    # ── Resolve initial prompt ───────────────────────────────────────────────
    raw = sys.argv[1] if len(sys.argv) > 1 else ""
    initial_prompt = re.sub(r"\s+", " ", raw).strip() or STATIC_FALLBACK_PROMPT

    # ── Build initial conversation ───────────────────────────────────────────
    messages: list = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": initial_prompt},
    ]

    print_separator("═")
    print(f"👤 You: {initial_prompt}")
    print_separator("═")

    # ── First generation ─────────────────────────────────────────────────────
    print("\n🤖 Assistant:\n")
    response = generate_response(model, tokenizer, messages, stream=True)
    display_response(response, streamed=True)
    messages.append({"role": "assistant", "content": response})

    # ── Interactive loop ─────────────────────────────────────────────────────
    print("💬 Chat mode activated. Type 'quit' / 'exit' / Ctrl-D to stop.\n")

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("👋 Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(messages, max_turns=MAX_HISTORY_TURNS)

        print("\n🤖 Assistant:\n")
        response = generate_response(model, tokenizer, messages, stream=True)
        display_response(response, streamed=True)
        messages.append({"role": "assistant", "content": response})

    print("\nSession ended.")


if __name__ == "__main__":
    main()