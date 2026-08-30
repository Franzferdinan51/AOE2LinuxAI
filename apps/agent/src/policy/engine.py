"""Match rules against the state, reserve resources, emit actions."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

import structlog

from .idle import distribute_idle, farm_bank_target
from .rules import Rule, load_rules

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .allocation import Allocation
    from .state import PolicyState

log = structlog.stdlib.get_logger()

_SPENDABLE: tuple[str, ...] = ("food", "wood", "gold", "stone")

_WOOD_BANK_MARGIN = 20


@lru_cache(maxsize=1)
def registry() -> tuple[Rule, ...]:
    """The shipped rules, highest weight first."""
    return load_rules()


def decide(
    entities: list[object],
    state: PolicyState,
    alarm: bool,
    rules: Sequence[Rule] | None = None,
    strategist_allocation: Allocation | None = None,
) -> list[dict[str, object]]:
    """Routine actions for this frame. Empty on alarm — the LLM owns combat."""
    if alarm:
        return []
    active = registry() if rules is None else tuple(rules)
    idle = distribute_idle(entities, state, wood_bank_target(state, active), strategist_allocation)
    return matched_actions(state, active) + idle


def matched_actions(state: PolicyState, rules: Sequence[Rule]) -> list[dict[str, object]]:
    """Rules whose trigger holds and whose cost the running balance can pay."""
    balance = {name: getattr(state, name) for name in _SPENDABLE}
    actions: list[dict[str, object]] = []
    for rule in rules:
        if not rule.enabled or not rule.matches(state):
            continue
        if not rule.is_fresh(state):
            log.info("policy_rule_stale", rule=rule.id, age_ms=round(state.age_ms))
            continue
        if not _can_afford(rule, balance):
            log.debug("policy_rule_unaffordable", rule=rule.id, cost=rule.cost, have=balance)
            continue
        for resource, amount in rule.cost.items():
            balance[resource] -= amount
        log.debug("policy_rule_fired", rule=rule.id, weight=rule.weight)
        actions.extend(rule.render(state))
    return actions


def wood_bank_target(state: PolicyState, rules: Sequence[Rule]) -> int | None:
    """Wood the idle rotation banks toward: the binding build goal, plus margin."""
    for rule in rules:
        cost = rule.cost.get("wood")
        if rule.bank_wood and rule.enabled and cost and rule.matches(state):
            return cost + _WOOD_BANK_MARGIN
    if "mill" in state.buildings_seen:
        return farm_bank_target()
    return None


def _can_afford(rule: Rule, balance: dict[str, int]) -> bool:
    return all(amount <= balance.get(resource, 0) for resource, amount in rule.cost.items())
