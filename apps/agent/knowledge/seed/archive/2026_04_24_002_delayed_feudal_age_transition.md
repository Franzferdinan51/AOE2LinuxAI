---
type: observation
title: delayed_feudal_age_transition
game_id: 2026-04-24-002
applies_when: 'age == "Dark Age" and turn > 30'
score_impact: negative
---

Castle Age gate was not researched until turn 32. The reactive rules now fire
the age-up at 500 food / 200 wood even without an explicit strategist signal.
