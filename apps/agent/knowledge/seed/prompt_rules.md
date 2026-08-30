# Prompt rules

Cross-game rules of the form:

    -id: short_identifier
    applies_when: "context clause"
    score_impact: positive | negative | neutral

The MemoryChain loader reads these from the frontmatter; the LLM uses them
as one-shot memories at the start of each game.

## Default rule set

- **no_phantom_evidence**: do not treat a building class as built without a
  wood-delta confirmation; detections are clues, not proof.

- **house_when_capped**: when the HUD shows pop-cap headroom ≤ 4, a house is
  the highest-priority action; the spent wood bankrolls the next farm cycle.

- **castle_age_gate**: two of {barracks, archery_range, stable, blacksmith,
  market} must be standing before the age-up is attempted.

- **first_four_on_food**: until turn 6, every villager should be on sheep or
  berries; a villager on wood that early costs ~10s of food gathering.

- **never_double_queue**: pressing 'q' twice in one turn delivers 2 villagers
  and doubles food spend; always verify the HUD population before re-pressing.
