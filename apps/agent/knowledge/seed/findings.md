# Findings

Hard-won observations from the run-history ledger. Each finding names a failure
pattern and the change that fixed it.

## 2026-04-22: housing stalls population growth at 35/35

When the LLM never builds houses, villagers queue behind a 4-slot TC and
production stalls around turn 25.

Fix: reactive tier auto-queues a house whenever pop-cap headroom ≤ 2.

## 2026-04-22: castle-age gate stays greyed out without two Feudal-Age buildings

The age-up panel reads "needs 2 of {barracks, archery_range, stable,
blacksmith, market}". Building all five is wasted; building zero leaves the
gate closed for the entire Feudal Age.

Fix: visualize the gate status in the per-turn context line so the LLM sees
the count and the names of what's standing.

## 2026-04-23: idle villagers sit on the TC idle icon for 50+ turns

A dark badge with villagers standing around the TC costs the LLM 30 actions
to dispatch one at a time. Worse, the dispatch camera move drops the
build-prereq evidence for a turn.

Fix: dispatch up to 6 per turn when the badge count is readable; fall back
to 1 per turn otherwise.

## 2026-04-24: phantom mill unlocks 14 outposts

A persistent misdetection of a mill (7 frames in a row) beat the 3-frame
sighting threshold and the LLM queued an outpost on every pop-cap headroom
for 14 turns.

Fix: detection sightings NEVER enter `buildings_confirmed`. Only wood-delta
settlement (the game's own ledger) or a visually verified placement does.
