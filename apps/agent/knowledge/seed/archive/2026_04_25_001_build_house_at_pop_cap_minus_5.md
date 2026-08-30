---
type: observation
title: build_house_at_pop_cap_minus_5
game_id: 2026-04-25-001
applies_when: 'population_cap - population <= 5'
score_impact: positive
---

The 5-headroom threshold lets the LLM build the house BEFORE the TC idle
queue fills, so the next villager can be queued immediately.
