---
type: observation
title: keyboard_input_sequence_ineffective
game_id: 2026-04-24-003
applies_when: 'building_key not in {"q","w","e","r","a","s","t"}'
score_impact: negative
---

A model attempted to press building_key='z' for a building; the game
rejected the press silently. Fix: validate building_key against the menu map
and reject out-of-menu keys at parse time.
