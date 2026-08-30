"""Memory storage + retrieval for the agent.

Persists per-game memories to markdown files in a directory; the agent
loads the most-relevant ones into context for each turn.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)


def extract_json_object(text: str) -> dict | None:
    """Best-effort extraction of the first top-level JSON object in `text`.

    LLMs sometimes wrap their JSON in prose or a fence; this lets both
    pass. Returns ``None`` if no JSON object is found.
    """
    if not text:
        return None
    # Strip ``` fences if present.
    cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.MULTILINE)
    cleaned = re.sub(r"```\s*$", "", cleaned, flags=re.MULTILINE)
    # Find the first '{' and parse object greedily (json.loads handles nested braces).
    start = cleaned.find("{")
    if start < 0:
        return None
    try:
        return json.loads(cleaned[start:])
    except json.JSONDecodeError:
        return None


def _read_memory_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_memory_title(frontmatter: dict, path: Path) -> str:
    """Pull a snake_case title from frontmatter, falling back to filename."""
    title = frontmatter.get("title")
    if isinstance(title, str) and title.strip():
        slug = re.sub(r"[^a-z0-9_]+", "_", title.lower()).strip("_")
        return slug or path.stem
    match = re.match(r"\d+_(.+)", path.stem)
    return match.group(1) if match else path.stem


@dataclass(frozen=True)
class _MemoryEntry:
    """One persisted memory fragment after YAML-frontmatter parsing."""

    rank: int
    created: str
    title: str
    applies_when: str
    content: str


@dataclass
class AgentMemory:
    """File-backed memory store.

    Each memory is one ``.md`` file with YAML-style frontmatter:

    ```
    ---
    type: strategy
    title: avoid_duplicate_house_spam
    game_id: 2026_04_25_001
    applies_when: "population >= 45"
    score_impact: negative
    created: 2026-04-25T19:32:11Z
    ---
    Free-text body…
    ```

    `memories_dir` defaults to ``apps/agent/knowledge/seed/memories/`` so a fresh
    clone sees the seed rules immediately.
    """

    memories_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2] / "knowledge" / "seed" / "memories"
    )

    _MAX_MEMORIES: int = 12
    _IMPACT_RANK: dict[str, int] = field(
        default_factory=lambda: {"negative": 0, "positive": 1, "neutral": 2}
    )

    def load_context(self, max_tokens: int = 800) -> str:
        """Load memory fragments into a first-person context string.

        Memories are ranked: negative `score_impact` first (traps to avoid),
        then positive (patterns to repeat), then neutral. Within each tier,
        most recently created first. Caps at `_MAX_MEMORIES` then trims by
        token budget.

        Args:
            max_tokens: Approximate token budget (1 token ~ 4 chars)

        Returns:
            Formatted string starting with `## Notes to Myself…`, or empty.
        """
        files = list(self.memories_dir.glob("*.md"))
        if not files:
            return ""

        entries: list[_MemoryEntry] = []
        for f in files:
            text = _read_memory_file(f)
            meta = self._parse_frontmatter(text)
            content = self._strip_frontmatter(text).strip()
            if not content:
                continue
            entries.append(
                _MemoryEntry(
                    rank=self._IMPACT_RANK.get(meta.get("score_impact", "neutral"), 2),
                    created=meta.get("created", ""),
                    title=_resolve_memory_title(meta, f),
                    applies_when=meta.get("applies_when", ""),
                    content=content,
                )
            )

        if not entries:
            return ""

        # Sort by impact tier ascending (negative first), then created descending
        # (newest first). created is an ISO 8601 string so lexicographic sort works.
        entries.sort(key=lambda e: e.created, reverse=True)
        entries.sort(key=lambda e: e.rank)
        ordered = entries[: self._MAX_MEMORIES]

        header = (
            "## Notes to Myself from Previous Games\n"
            "Each bullet is a rule I wrote for myself after finishing a game, based "
            "on what actually happened.\n"
            "\n"
            "**Precedence: when a memory rule conflicts with a rule in core.md or "
            "the age-specific section, follow the MEMORY.** Memories reflect "
            "concrete evidence from my own games; the defaults are pre-game "
            "heuristics. If two memories conflict, prefer the one whose "
            "`(when: …)` trigger is more specific or matches my current state "
            "more tightly.\n"
            "\n"
            "I should apply any rule whose trigger matches my current state.\n"
        )
        char_budget = max_tokens * 4
        lines: list[str] = []
        total_chars = len(header)

        for entry in ordered:
            # First line of content keeps the bullet compact; multi-sentence
            # content is preserved verbatim after the trigger prefix.
            when_prefix = (
                f"(when: {entry.applies_when}) "
                if entry.applies_when and entry.applies_when != "any"
                else ""
            )
            # `[title]` makes the snake_case identifier visible to the model so it
            # can emit the `[applied: title]` reasoning prefix per prompts/core.md.
            line = f"- [{entry.title}] {when_prefix}{entry.content}"
            if total_chars + len(line) + 1 > char_budget:
                break
            lines.append(line)
            total_chars += len(line) + 1

        if not lines:
            return ""

        return header + "\n".join(lines)

    def list_memories(self) -> list[dict]:
        """List all memory fragments with metadata."""
        result = []
        for f in sorted(self.memories_dir.glob("*.md")):
            text = _read_memory_file(f)
            meta = self._parse_frontmatter(text)
            content = self._strip_frontmatter(text).strip()
            title = _resolve_memory_title(meta, f)
            result.append(
                {
                    "file": f.name,
                    "title": title,
                    "type": meta.get("type", "unknown"),
                    "game_id": meta.get("game_id", "unknown"),
                    "applies_when": meta.get("applies_when", ""),
                    "score_impact": meta.get("score_impact", "neutral"),
                    "content": content,
                }
            )
        return result

    def _build_game_summary(self, memory: AgentMemory, score: GameScore) -> str:
        """Build a text summary of the game for the extraction LLM."""
        parts = []

        # Metrics
        metrics = memory.get_metrics_snapshot()
        parts.append(f"Game Result: {metrics['game_end_reason'] or 'unknown'}")
        parts.append(
            f"Score: {score.composite:.4f} (age={score.age:.2f}, "
            f"age_speed={score.age_speed:.2f}, economy={score.economy:.2f}, "
            f"actions={score.action_success:.2f}, survival={score.survival:.2f})"
        )
        parts.append(f"Duration: {metrics['survival_time']:.0f}s, Turns: {metrics['turn_count']}")
        parts.append(
            f"Peak Population: {metrics['peak_population']}, Highest Age: {metrics['highest_age']}"
        )
        parts.append("")

        # Turn history
        turns = list(memory.working_memory)
        if not turns:
            return ""

        parts.append("Turn-by-turn summary (last 10 turns):")
        for t in turns:
            action_summary = ", ".join(
                f"{a.get('type', '?')}({a.get('key', a.get('target_id', ''))})"
                for a in t.actions[:4]
            )
            parts.append(f"  Turn {t.iteration}: {t.reasoning[:150]}")
            parts.append(f"    Actions: {action_summary}")
            if t.observed_resources:
                parts.append(f"    Resources: {t.observed_resources}")

        return "\n".join(parts)

    def _parse_observations(self, text: str) -> list[dict[str, object]]:
        """Parse LLM response into observation dicts.

        Drops entries with empty content (was producing 0-byte files —
        see exp_0011's `006_missing_feudal_age_target.md`).
        """
        data = extract_json_object(text)
        if data is None:
            log.warning("memory_parse_failed", text=text[:200])
            return []

        raw_value = data.get("observations")
        raw: list[object] = raw_value if isinstance(raw_value, list) else []
        valid = [obs for obs in raw if isinstance(obs, dict) and (obs.get("content") or "").strip()]
        if len(valid) < len(raw):
            log.warning("memory_parse_dropped_empty", dropped=len(raw) - len(valid))
        return valid

    def _next_file_number(self) -> int:
        """Get the next available file number."""
        existing = list(self.memories_dir.glob("*.md"))
        if not existing:
            return 1
        numbers = []
        for f in existing:
            match = re.match(r"(\d+)_", f.name)
            if match:
                numbers.append(int(match.group(1)))
        return max(numbers, default=0) + 1

    def _existing_titles(self) -> set[str]:
        """Return the set of sanitized titles already saved on disk.

        Reads from frontmatter `title:` first, falls back to filename suffix.
        """
        titles: set[str] = set()
        for f in self.memories_dir.glob("*.md"):
            meta = self._parse_frontmatter(_read_memory_file(f))
            title = meta.get("title")
            if not title:
                # Filename pattern: NNN_<title>.md
                match = re.match(r"\d+_(.+)\.md$", f.name)
                if match:
                    title = match.group(1)
            if title:
                titles.add(title)
        return titles

    def _save_memory(self, obs: dict, game_id: str, file_num: int) -> Path | None:
        """Save a single observation as a markdown file."""
        mem_type = obs.get("type", "strategy")
        title = obs.get("title", "observation")
        content = (obs.get("content") or "").strip()
        score_impact = obs.get("score_impact", "neutral")
        applies_when = (obs.get("applies_when") or "any").strip()

        if not content:
            return None

        # Sanitize title for filename
        safe_title = re.sub(r"[^a-z0-9_]", "_", title.lower())[:50]
        filename = f"{file_num:03d}_{safe_title}.md"
        path = self.memories_dir / filename

        now = datetime.now(UTC).isoformat(timespec="seconds")
        file_content = f"""---
type: {mem_type}
title: {safe_title}
game_id: {game_id}
applies_when: {applies_when}
score_impact: {score_impact}
created: {now}
---

{content}
"""
        path.write_text(file_content, encoding="utf-8")
        return path

    def _read_memory(self, path: Path) -> str | None:
        """Read a memory file and return just the content (no frontmatter)."""
        text = _read_memory_file(path)
        content = self._strip_frontmatter(text).strip()
        return content if content else None

    def _parse_frontmatter(self, text: str) -> dict:
        """Parse YAML-like frontmatter from a memory file."""
        match = re.match(r"^---\n(.+?)\n---", text, re.DOTALL)
        if not match:
            return {}
        meta = {}
        for line in match.group(1).split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip()] = value.strip()
        return meta

    def _strip_frontmatter(self, text: str) -> str:
        """Remove frontmatter from text."""
        match = re.match(r"^---\n.+?\n---\n?", text, re.DOTALL)
        if match:
            return text[match.end() :]
        return text
