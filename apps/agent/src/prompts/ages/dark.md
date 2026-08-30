# Dark Age — 100% Economy

Your only goal in Dark Age is to grow your economy as fast as possible. No military, no fighting.

## Mill + 3 farms by turn 5 (PROACTIVE FOOD RULE)

**Plant farms BEFORE the food crisis, not after.** Farms take ~60 s to start producing — if you wait until food < 50, the crisis is already locked in.

**As soon as you have 100 wood AND 4+ villagers gathering food:**
1. If NO `mill` is in the Detected Entities list, build ONE Mill (`build` with `building_key="w"`) next to a berry bush. If a `mill` is already detected, skip this — one Mill is all you ever need.
2. Plant 3 farms (`build` × 3 with `building_key="a"`) adjacent to the Mill or TC.

By turn 5–8, NOT turn 15+. Sheep deplete around turn 10–12; the farm pipeline must already be running.

## Dark Age Checklist (in addition to universal checklist)

**Villager queuing:** the reactive tier queues villagers up to the Dark Age **order target of 30** automatically, then banks food — you rarely need to queue by hand.
- **Fewer than 30 villagers ordered**: queuing is fine. If 150+ food, 2–3 is fine.
- **30 ordered**: STOP queuing — save food for Feudal (500).

**Villager allocation:** 6–8 on food, 3–4 on wood initially. Never have 0 food gatherers.

**Scout:** Enable Auto Scout ONCE. Press `,` then `G`. The scout will explore on its own.

**Lumber Camp:** Build by turn 10–15. 2+ villagers on wood without one? Build (`build` with `building_key="r"`, 100 wood). Counts as one of the 2 prereq buildings for Feudal.

**Berries:** `berry_bush` detected but no Mill nearby? Build a Mill (`build` with `building_key="w"`) and send 3–4 villagers via `send_villager target_class=berry_bush`.

**Feudal Age transition:** see the **Age-up Gate** in core.md. The qualifying prereq pair is **Mill AND Lumber Camp**.

**Mill + Farms emergency:** NO sheep AND no berry_bush in entity list AND food < 100? → P10 EMERGENCY. Drop everything else and get farms running this turn.

**One Mill is enough — decide from the Detected Entities list, never from habit:**
- **`mill` IS in the Detected Entities list** → you already have your food drop-off. Do NOT build another Mill. Build **farms only**.
- **NO `mill` in the Detected Entities list** → build exactly ONE Mill (`building_key="w"`) this turn, then farms around it.

**Sanity check the mill detection (avoid a house-as-mill trap).** If a `mill` appears in detections BUT the strategist reasoning says *no mill visible*, treat the `mill` as a misdetection and **build a real Mill**.

## Food Economy

**Gathering order:** sheep → berries (build Mill near berries) → farms (build Mill anywhere, then 1 farm per food villager).

**Notes:**
- **NEVER right-click a boar.** Boars fight back and kill villagers. Ignore them; use sheep → berries → farms.
- Each farm supports only 1 villager. Don't double-up.
- If `target_class: "sheep"` fails once, sheep are not detected — build Mill + farms instead, do not retry sheep.

## Emergency: Under Attack

**In Dark Age: NEVER press B (town bell) or T (garrison) — no exceptions.** The TC's auto-arrows + economy continuity beats garrisoning.

**NEVER build Towers or any defensive/military building in Dark Age.** A starving economy loses far faster than enemy harassment does.

**Strategist's `alarm` flag in Dark Age: ignore it.** Continue your build order.

**Accidentally garrisoned?** Press H → V (All Back to Work) to release everyone.

## Build Menu Restriction

In Dark Age, ONLY use the Q build menu (economic: House, Mill, Mining Camp, Lumber Camp, Farm). Do NOT touch W (military) or V (advanced) menus.
