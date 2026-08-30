---
type: observation
title: missing_feudal_age_target
game_id: 2026-04-24-006
applies_when: 'food >= 500 and wood >= 200'
score_impact: negative
---

Memory files were being saved with empty content (0-byte) when the
extraction LLM returned all-zero observations. Fix: dropped empty observations
at parse time.
