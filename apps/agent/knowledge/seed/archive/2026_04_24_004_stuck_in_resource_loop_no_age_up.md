---
type: observation
title: stuck_in_resource_loop_no_age_up
game_id: 2026-04-24-004
applies_when: 'food > 800 and wood > 200 and age == "Dark Age"'
score_impact: negative
---

Model kept building farms instead of researching. Fix: the policy tier fires
the age-up research when both resources are met, regardless of model output.
