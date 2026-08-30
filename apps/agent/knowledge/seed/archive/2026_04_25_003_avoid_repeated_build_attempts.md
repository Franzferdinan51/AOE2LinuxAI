---
type: observation
title: avoid_repeated_build_attempts
game_id: 2026-04-25-003
applies_when: 'building_already_pending'
score_impact: negative
---

The LLM kept re-issuing the same build action every turn despite pending
settlement. Fix: `pending_placement_counts()` is now in the context line so
the LLM sees and respects it.
