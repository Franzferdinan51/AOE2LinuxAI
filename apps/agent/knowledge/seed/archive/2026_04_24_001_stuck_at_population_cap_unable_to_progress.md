---
type: observation
title: stuck_at_population_cap_unable_to_progress
game_id: 2026-04-24-001
applies_when: 'population == population_cap and pop_cap < 60'
score_impact: negative
---

The agent repeatedly queued villagers without building houses, leading to
the TC queue backing up and food income stalling around turn 25. Fix: build a
house whenever pop-cap headroom drops below 5.
