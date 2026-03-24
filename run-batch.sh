#!/usr/bin/env bash

set -euo pipefail

# 1. Standard arrays for looping and random picking
ROLES=("Assassin" "Fighter" "Tank" "Support" "Mage")
ELEMENTS=("Earth" "Fire" "Ice" "Thunder" "Water" "Wind" "Plant" "Light" "Arcane")

# 2. Associative Arrays (Dictionaries) for specific descriptions
declare -A ROLE_DESC
ROLE_DESC=(
    ["Assassin"]="agile, stealthy, deadly"
    ["Fighter"]="aggressive, powerful melee warrior"
    ["Tank"]="massive, armored, protective"
    ["Support"]="mystical, nurturing, radiant"
    ["Mage"]="mysterious, arcane, intelligent"
)

declare -A ELEM_DESC
ELEM_DESC=(
    ["Earth"]="Moss-covered slate armor with deep red-maroon crystal clusters. Reinforced with heavy bronze plates and glowing maroon ley-line etchings."
    ["Fire"]="Charred carbon-fiber armor constantly shedding glowing embers. Instead of veins, include open furnace-ports on the chest emitting a blinding white-blue heat and thick black smoke trails."
    ["Ice"]="Armor of fractured ice and frosted metal with sharp crystalline edges. Deep sapphire-blue cracks glow with cold energy."
    ["Thunder"]="Carbon-fiber armor in indigo and slate gray, crossed by translucent conduits carrying glowing purple lightning."
    ["Water"]="Polished abalone shell and flowing liquid-metal armor. Limbs partially formed from swirling water, revealing a cyan energy core."
    ["Wind"]="Sleek silver-white armor with curved aerodynamic plates floating slightly from the body, etched with glowing turquoise runes."
    ["Plant"]="Body textured like bark, vines, and moss. Emerald sap veins glow beneath the surface, with leaves or thorn-like growths."
    ["Light"]="Radiant armor of polished sun-steel and gold, engraved with sacred geometric patterns and surrounded by floating halos of light."
    ["Arcane"]="Dark crimson alloy armor with floating plates over a bodysuit of glowing purple circuitry, powered by exposed magenta energy cores."
)
FORCED_ELEMENT="${1:-}"

declare -A ELEM_SPECIES
ELEM_SPECIES=(
    ["Earth"]="Stone Golems|Trolls|Dwarves|Earth Elementals|Burrowers|Crystal Behemoths|Mud Serpents|Boulder Giants|Rootbound Elves|Quartz Spirits|Pebble Swarms|Terracotta Warriors"
    ["Fire"]="Fire Dragons|Phoenixes|Flame Imps|Lava Elementals|Hellhounds|Ember Sprites|Volcanic Trolls|Inferno Demons|Charcoal Goblins|Blaze Serpents|Ashen Wraiths|Magma Giants"
    ["Ice"]="Frost Giants|Ice Dragons|Yetis|Snow Elementals|Polar Wraiths|Glacial Bears|Freeze Sprites|Arctic Elves|Blizzard Hounds|Crystal Serpents|Tundra Trolls|Permafrost Golems"
    ["Thunder"]="Thunderbirds|Storm Giants|Lightning Elementals|Electric Eels (Aerial Variant)|Volt Imps|Thunder Wolves|Plasma Demons|Bolt Serpents|Storm Elves|Arc Behemoths|Static Wraiths|Cyclone Goblins"
    ["Water"]="Mermaids|Krakens|Water Elementals|Sea Serpents|Aquatic Elves|Tidal Giants|Bubble Sprites|Depth Demons|River Trolls|Wave Golems|Mist Wraiths|Coral Behemoths"
    ["Wind"]="Harpies|Griffins|Air Elementals|Sylphs|Storm Eagles|Zephyr Imps|Cyclone Giants|Breeze Elves|Vortex Serpents|Gale Hounds|Draft Wraiths|Tempest Goblins"
    ["Plant"]="Ents|Dryads|Treants|Vine Serpents|Flower Sprites|Thorn Giants|Moss Trolls|Petal Elves|Root Behemoths|Leaf Demons|Sap Wraiths|Bloom Goblins"
    ["Light"]="Angels|Unicorns|Light Elementals|Radiant Fairies|Solar Eagles|Luminous Elves|Prism Serpents|Dawn Giants|Glow Imps|Halo Hounds|Luminary Wraiths|Beacon Golems"
    ["Arcane"]="Arcane Elementals|Liches|Mana Beasts|Spellweavers|Ether Sprites|Rune Giants|Mystic Elves|Void Serpents|Enchantment Imps|Portal Hounds|Astral Wraiths|Glyph Goblins"
)

for role in "${ROLES[@]}"; do
    # Use provided element or pick random
    if [[ -n "$FORCED_ELEMENT" ]]; then
        element="$FORCED_ELEMENT"
    else
        element=${ELEMENTS[$RANDOM % ${#ELEMENTS[@]}]}
    fi

    IFS='|' read -ra current_species <<< "${ELEM_SPECIES[$element]}"
    random_species="${current_species[$RANDOM % ${#current_species[@]}]}"

    echo "========================================"
    echo "⚔️ Generating $element $role Kaimon from $random_species species"
    echo "========================================"

    # Construct the laser-focused prompt
    PROMPT="Create a majestic Kaimon from the Kiooverse.

Kaimons are mythical creatures born from magical eggs. 
They resemble animals but evolved into elegant elemental beings.

Design a $element element Kaimon inspired by $random_species species.

Class archetype: $role.
The creature should express the personality of a $role class: ${ROLE_DESC[$role]}.

The creature should look like the element behaviour:
${ELEM_DESC[$element]}

The $element element must strongly influence its design, colors, aura, and abilities.
The pose is a dynamic offensive guard stance, ready for battle.

High fantasy creature design, highly detailed, majestic, beautiful, powerful, unique species of the Kiooverse."

    # echo $PROMPT
    # Pass the clean prompt to the Python script
    ./run.sh "$PROMPT"

done

echo "🎉 All Kaimon classes generated!"
