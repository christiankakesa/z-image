#!/usr/bin/env bash

set -euo pipefail

ROLES=("Assassin" "Fighter" "Tank" "Support" "Mage")

ELEMENTS=("Earth" "Fire" "Ice" "Thunder" "Water" "Wind" "Plant" "Light" "Arcane")

ANIMALS=("wolf" "tiger" "panther" "owl" "serpent" "dragon" "stag" "fox" "lion" "mantis" "raven" "bear")

BODIES=("quadruped" "biped" "winged" "serpentine" "centaur-like")

FORCED_ELEMENT="${1:-}"

for role in "${ROLES[@]}"; do

    # Use provided element or pick random
    if [[ -n "$FORCED_ELEMENT" ]]; then
        element="$FORCED_ELEMENT"
    else
        element=${ELEMENTS[$RANDOM % ${#ELEMENTS[@]}]}
    fi
    
    # Random animal inspiration
    animal=${ANIMALS[$RANDOM % ${#ANIMALS[@]}]}
    
    # Random body type
    body=${BODIES[$RANDOM % ${#BODIES[@]}]}

    echo "========================================"
    echo "⚔️ Generating $element $role Kaimon"
    echo "========================================"

    ./run.sh "
Create a majestic Kaimon from the Kiooverse.

Kaimons are mythical creatures born from magical eggs. 
They resemble animals but evolved into elegant elemental beings.

Design a $element element Kaimon inspired by a $animal.
Body type: $body.

Class archetype: $role.

The creature should express the personality of a $role class:
- Assassin: agile, stealthy, deadly
- Fighter: aggressive, powerful melee warrior
- Tank: massive, armored, protective
- Support: mystical, nurturing, radiant
- Mage: mysterious, arcane, intelligent

The creature should look like the element behaviour:
- Earth: Matte rock texture and mineral veins. Armor reinforced with gold filigree and glowing ochre tectonic cracks, radiating warm orange energy.
- Fire: Obsidian-black armor with thermal vents and glowing magma veins—thin orange-red lines pulsing like molten lava beneath the surface.
- Ice: Armor of fractured ice and frosted metal with sharp crystalline edges. Deep sapphire-blue cracks glow with cold energy.
- Thunder: Carbon-fiber armor in indigo and slate gray, crossed by translucent conduits carrying glowing purple lightning.
- Water: Polished abalone shell and flowing liquid-metal armor. Limbs partially formed from swirling water, revealing a cyan energy core.
- Wind: Sleek silver-white armor with curved aerodynamic plates floating slightly from the body, etched with glowing turquoise runes.
- Plant: Body textured like bark, vines, and moss. Emerald sap veins glow beneath the surface, with leaves or thorn-like growths.
- Light: Radiant armor of polished sun-steel and gold, engraved with sacred geometric patterns and surrounded by floating halos of light.
- Arcane: Dark crimson alloy armor with floating plates over a bodysuit of glowing purple circuitry, powered by exposed magenta energy cores.

The $element element must strongly influence its design, colors, aura, and abilities.

The pose is a dynamic offensive guard stance, ready for battle.

High fantasy creature design, highly detailed, majestic, beautiful, powerful, unique species of the Kiooverse.
"

done

echo "🎉 All Kaimon classes generated!"
