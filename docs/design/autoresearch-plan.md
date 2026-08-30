# Autoresearch for AoE2 Agent — Continuous Improvement Plan

**Status:** **PARTIALLY SHIPPED** — Phase 0 + Phase 1 (prompt-mutation loop with git-revert + memory chain) live in `autoresearch/`. Phases 2–5 unbuilt. Frozen historical plan; for current state see [Part 8 — Autoresearch](../part8-autoresearch/22-autoresearch-overview.md).
**Original location:** repo root `AUTORESEARCH_PLAN.md` (moved 2026-05-24).

> Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch): let an LLM autonomously experiment in a tight loop — modify → evaluate → keep/revert → repeat. This plan adapts that pattern to continuously improve the AoE2 game-playing agent.

---

## Table of Contents

1. [Background & Motivation](#1-background--motivation)
2. [Current Agent Architecture](#2-current-agent-architecture)
3. [Autoresearch Concept](#3-autoresearch-concept-how-it-maps-to-aoe2)
4. [Bug Fixes (prerequisite)](#4-bug-fixes-prerequisite)
5. [Phase 0: Foundation (COMPLETED)](#5-phase-0-foundation-completed)
6. [Phase 1: Prompt Optimization Loop](#6-phase-1-prompt-optimization-loop)
7. [Phase 2: Context Tuning + Strategy Mining](#7-phase-2-context-tuning--strategy-mining)
8. [Phase 3: Automated Game Restart](#8-phase-3-automated-game-restart)
9. [Phase 4: Detection Active Learning](#9-phase-4-detection-active-learning)
10. [Phase 5: Training Pipeline Improvements](#10-phase-5-training-pipeline-improvements)
11. [Scoring System](#11-scoring-system)
12. [File Reference](#12-file-reference)
13. [Cost Estimates](#13-cost-estimates)

---

## 1. Background & Motivation

### The Problem

The AoE2 agent can play the game — it captures screenshots, perceives them locally (YOLO entity detection + OCR of the resource bar), sends that as text to Claude, receives actions, and executes them via pyautogui. But **it never learns from its gameplay**. Every game starts from the same system prompt with the same strategy. There is no feedback loop from game outcomes back to the agent's behavior.

### The Autoresearch Pattern (Karpathy)

Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) demonstrates a powerful pattern for autonomous improvement:

1. An LLM agent has **one file** it can modify (`train.py`)
2. It proposes a change and commits it
3. It runs a **fixed-budget evaluation** (5 minutes of GPU training)
4. It measures **one clear metric** (`val_bpb` — validation bits per byte)
5. If the metric improved → keep the commit. If worse → `git reset`
6. Loop forever (~100 experiments overnight)

**Key insight**: The magic is in the constraints — one file, one metric, fixed budget, git-based accept/reject.

### How This Maps to AoE2

| Autoresearch | AoE2 Agent |
|---|---|
| `train.py` (file to modify) | `prompts/system.md` (system prompt) |
| `val_bpb` (metric) | Composite game score (survival + population + age + economy) |
| 5-min GPU training | 20-min game vs Easiest AI |
| LLM proposes code change | LLM proposes prompt change |
| `git reset` on failure | `git checkout -- prompts/system.md` |
| ~100 experiments/night | ~24 experiments/night (games are slower) |

---

## 2. Current Agent Architecture

```
Screenshot → YOLO Detection (60 classes) + resource-bar OCR → Entity + Resource Context (text) → Claude → JSON Actions → pyautogui
     ↑                                                                                          |
     └─────────────────────────────── 2s loop delay ─────────────────────────────────┘
```

### Key Files

| File | Purpose |
|---|---|
| `gameplay_agent/game_loop.py` | Core capture→detect→think→act cycle (2-second loop) |
| `gameplay_agent/providers/executor_provider.py` | Sends screenshot + context to Claude, parses JSON response |
| `gameplay_agent/memory.py` | Turn history, game state tracking, cumulative metrics |
| `gameplay_agent/models.py` | Pydantic models for actions and observations |
| `gameplay_agent/executor.py` | Translates actions to pyautogui calls |
| `gameplay_agent/screen.py` | Screenshot capture via mss |
| `gameplay_agent/window.py` | AoE2 window detection and focus |
| `prompts/system.md` | System prompt with game rules, hotkeys, output format |
| `detection/inference/detector.py` | YOLO26n entity detection (60 classes; current served model v9, real F1 ≈ 0.67 single-pass @1280 — the metric of record, not synthetic mAP) |

### Data Flow Per Turn

1. `capture_screenshot()` → JPEG bytes + dimensions
2. `detector.detect(screenshot)` → list of `DetectedEntity` (id, class, bbox, confidence)
3. `memory.get_context_for_llm()` → game state + recent turns as text
4. Entity context formatted as `sheep_0: sheep at (640,380) [92%]`
5. `provider.get_actions(screenshot, context, width, height)` → Claude API call
6. Response parsed via `messages.parse()` into `LLMResponse` (Pydantic model)
7. `memory.create_turn(reasoning, actions, observations)` → updates game state
8. `execute_actions(actions)` → pyautogui clicks/keypresses, returns success_count

---

## 3. Autoresearch Concept: How It Maps to AoE2

We define **four parallel improvement loops**, each with its own "file to modify", "metric to optimize", and "evaluation budget":

| Loop | What Gets Modified | Metric | Eval Time | Cadence |
|---|---|---|---|---|
| 1. Prompt Optimization | `prompts/system.md` | Composite game score | 20 min/game | Every game |
| 2. Strategy Mining | `data/strategy.db` → injected context | Win rate | 0 (piggybacks) | Every 3 games |
| 3. Context Tuning | `autoresearch/context_config.yaml` | Action success rate | 2 min/test | Between games |
| 4. Detection Learning | YOLO model weights | mAP50 + action success | 2 hrs + 3 games | Weekly |

---

## 4. Bug Fixes (prerequisite)

> Absorbed from IMPROVEMENT_PLAN.md Part 1. These are standalone bug fixes that should be addressed before or alongside autoresearch work.

> **Status: ALL DONE.** All items below have been implemented:
> - 4.1 Entity ID persistence (IoU tracking) — `_assign_persistent_ids()` in `detector.py`
> - 4.2 NMS for PyTorch — unified `_nms()` in `detect()` for all backends
> - 4.3 Window offset per-action — re-fetch in `execute_action()` instead of `execute_actions()`
> - 4.4 Debug print cleanup — replaced with `logger.debug()` calls
> - 4.5 Action verification — pre/post detection comparison in `game_loop.py`
> - Additionally: structured output via `messages.parse()` replaced custom JSON parsing in `executor_provider.py`

### 4.1 Entity ID Persistence — IoU-Based Tracking ✅

**Severity**: HIGH
**File**: `detection/inference/detector.py`

**Problem**: `_reset_counters()` clears all entity ID counters at the start of every detection cycle. Entity IDs like `sheep_0` are regenerated from scratch each frame. The LLM targets `sheep_0` in turn N, but by turn N+1 a completely different sheep may be assigned `sheep_0`.

**Fix**: Add `_previous_detections` cache. After each detection cycle, match new detections to previous ones by IoU overlap. If IoU > 0.4, reuse the old entity ID. If no match, assign a new ID with an incrementing global counter (never reset).

```python
# New fields in EntityDetector.__init__():
self._previous_detections: list[DetectedEntity] = []
self._global_id_counter: int = 0

def _assign_persistent_ids(self, new_detections: list[DetectedEntity]) -> list[DetectedEntity]:
    """Match new detections to previous frame by IoU, preserving IDs."""
    used_prev = set()
    result = []
    for new_det in new_detections:
        best_iou, best_prev = 0.0, None
        for i, prev_det in enumerate(self._previous_detections):
            if i in used_prev or prev_det.class_name != new_det.class_name:
                continue
            iou = self._compute_iou(new_det.bbox, prev_det.bbox)
            if iou > best_iou:
                best_iou, best_prev = iou, (i, prev_det)
        if best_prev and best_iou > 0.4:
            used_prev.add(best_prev[0])
            new_det.id = best_prev[1].id
        else:
            new_det.id = f"{new_det.class_name}_{self._global_id_counter}"
            self._global_id_counter += 1
        result.append(new_det)
    self._previous_detections = result
    return result
```

Call `_assign_persistent_ids()` at the end of `detect()` instead of `_reset_counters()` at the beginning.

### 4.2 NMS Missing in PyTorch Backend ✅

**Severity**: MEDIUM
**File**: `detection/inference/detector.py`

**Problem**: `_nms()` method defined but never called for the PyTorch inference path. Only the ONNX path applies NMS. This means PyTorch detections can include duplicate overlapping boxes.

**Fix**: After the PyTorch results loop, add:
```python
entities = self._nms(entities, iou_threshold=0.5)
```

### 4.3 Window Offset Race Condition ✅

**Severity**: MEDIUM
**File**: `gameplay_agent/executor.py`

**Problem**: Window rect is fetched once at the start of action batch execution. If the game window moves during the batch, all subsequent coordinate translations are wrong.

**Fix**: Re-fetch window rect before each individual action:
```python
window_rect = self.window.get_game_window_rect()  # Fresh fetch per action
```

### 4.4 ONNX Debug Print Spam ✅

**Severity**: LOW
**File**: `detection/inference/detector.py`

**Problem**: Multiple `print("DEBUG:...")` statements left in production code.

**Fix**: Replace all with `log.debug()` using the existing structlog logger.

### 4.5 Action Verification Enhancement ✅

**Severity**: MEDIUM
**Files**: `gameplay_agent/game_loop.py`, `gameplay_agent/memory.py`

**Current state**: Phase 0 tracks `success_count` from `execute_actions()` return value. This is a basic count — it doesn't tell the LLM *what* succeeded or failed.

**Enhancement**: Capture a post-action screenshot, compare pre/post entity states, and inject verification text into the next turn's LLM context:

```python
# After execute_actions():
post_screenshot = capture_screenshot()
post_entities = detector.detect(post_screenshot) if detector else []

verification = _verify_actions(pre_entities, post_entities, actions)
memory.last_verification = verification

# In memory.get_context_for_llm():
if self.last_verification:
    parts.append(f"## Last Turn Results\n{self.last_verification}")
```

Verification text example:
```
- Sent villager_2 to gold_mine_0: SUCCESS (villager moved 45px toward gold)
- Built house (press Q): UNCERTAIN (no new house detected yet)
```

---

## 5. Phase 0: Foundation (COMPLETED)

> Status: **DONE**. All items below are implemented and tested.

### What Was Built

#### 4.1 Game State Detection (`gameplay_agent/models.py`)

Added `game_state` field to the `Observations` Pydantic model:

```python
class Observations(BaseModel):
    resources: dict[str, int] = Field(default_factory=dict)
    population: str = ""
    age: str = ""
    idle_tc: bool = False
    under_attack: bool = False
    game_state: Literal["playing", "victory", "defeat", "menu"] = "playing"  # NEW
    events: list[str] = Field(default_factory=list)
```

The LLM reports game state in every response. The game loop checks it and stops on victory/defeat.

**Design decision**: We use the LLM's reported game state rather than template matching or pixel heuristics — the executor already emits an observation (resources, population, age, events) every turn, so a victory/defeat signal rides the same channel without extra perception code. (Perception is local: YOLO entities + resource-bar OCR as text; no image is sent to the model.)

#### 4.2 Cumulative Metrics (`gameplay_agent/memory.py`)

Added to `AgentMemory.__init__()`:
```python
# Cumulative metrics for autoresearch scoring
self.total_food_gathered: int = 0      # Highest food value observed
self.peak_population: int = 0          # Highest population reached
self.total_actions: int = 0            # All actions sent to executor
self.successful_actions: int = 0       # Actions that succeeded
self.highest_age: str = "Dark Age"     # Best age advancement
self.game_start_time: datetime | None = None  # Set on first turn
self.game_end_reason: str = ""         # "victory", "defeat", "timeout", "interrupted"
```

Updated in these methods:
- `add_turn()` → starts timer, counts actions, tracks food
- `update_from_observations()` → tracks peak population, highest age
- `record_action_results(success_count, total)` → increments successful_actions
- `get_metrics_snapshot()` → returns dict of all metrics for scoring
- `reset()` → clears all counters for new game

#### 4.3 Game-Over Detection + Time Budget (`gameplay_agent/game_loop.py`)

The `game_loop()` function was updated:

```python
async def game_loop(
    provider: BaseLLMProvider,
    max_iterations: int | None = None,
    memory: AgentMemory | None = None,
    use_detection: bool = True,
    time_budget: float | None = None,    # NEW: seconds limit
) -> AgentMemory:                        # NEW: returns memory with metrics
```

After each LLM response, two new checks:

```python
# 5b. Check for game-over via LLM observations
game_state = observations.get("game_state", "playing")
if game_state in ("victory", "defeat"):
    memory.game_end_reason = game_state
    break

# 5c. Check time budget
if time_budget and memory.get_game_duration_seconds() >= time_budget:
    memory.game_end_reason = "timeout"
    break
```

Action success is tracked after execution:

```python
if actions:
    success_count = await execute_actions(actions)
    memory.record_action_results(success_count, len(actions))
```

On exit (including errors/interrupts), final metrics are logged and memory is returned.

#### 4.4 Composite Scoring (`autoresearch/metrics.py`)

```python
@dataclass
class GameScore:
    composite: float      # 0.0 - 1.0 overall score
    survival: float       # component: time survived
    population: float     # component: peak pop
    age: float           # component: age advancement
    economy: float       # component: food gathered
    action_success: float # component: action success rate
    raw_metrics: dict    # original metrics snapshot

def compute_score(metrics: dict) -> GameScore:
    """Converts AgentMemory.get_metrics_snapshot() into a GameScore."""
```

**Weights** (must sum to 1.0):
| Component | Weight | Normalization Cap |
|---|---|---|
| Survival time | 0.30 | 1200 seconds (20 min) |
| Peak population | 0.25 | 50 villagers |
| Age advancement | 0.20 | Dark=0, Feudal=0.33, Castle=0.66, Imperial=1.0 |
| Economy (food) | 0.15 | 5000 food gathered |
| Action success rate | 0.10 | success_count / total_actions |

#### 4.5 Experiment Ledger (`autoresearch/experiment_log.py`)

TSV file at `experiments/results.tsv` tracking all experiments:

```
experiment_id  timestamp                loop    change_description  composite_score  survival  population  age  economy  action_success  game_end_reason  turn_count  accepted  git_sha
exp_0001       2026-03-15T22:00:00+00:00  manual  baseline          0.4500           0.8000    0.3000      0.0  0.2000   0.5000          timeout          450         true      abc1234
```

**Key functions**:
- `log_experiment(experiment_id, loop, description, score, accepted, git_sha)` → appends row
- `get_recent_experiments(n=5)` → reads last N experiments as list of dicts
- `get_best_score(loop=None)` → best composite score from accepted experiments
- `get_next_experiment_id()` → auto-increments `exp_NNNN`
- `get_git_sha()` → current short SHA

#### 4.6 Game Runner (`autoresearch/game_runner.py`)

CLI wrapper that runs a game and logs results:

```bash
# Run a 20-minute game with metrics collection
python -m autoresearch.game_runner --time-budget 1200 --description "baseline"

# Run with turn limit instead
python -m autoresearch.game_runner --max-iterations 500

# Specify experiment ID
python -m autoresearch.game_runner --experiment-id exp_0001 --description "added sheep priority"
```

**Key functions**:
- `run_game(time_budget, max_iterations, use_detection)` → runs game, returns `{metrics, score}`
- `run_and_log(experiment_id, loop, description, ...)` → runs game + logs to TSV

#### 4.7 System Prompt Update (`prompts/system.md`)

Added `game_state` to the output format example and a new section:

```markdown
## Game State Detection
Set `game_state` in observations:
- `"playing"` — normal gameplay (default)
- `"victory"` — you see a victory screen or "You are victorious" message
- `"defeat"` — you see a defeat screen or "You have been defeated" message
- `"menu"` — you see the main menu, loading screen, or lobby (not in a game)
```

#### 4.8 Configuration (`autoresearch/config.yaml`)

```yaml
game:
  time_budget: 1200        # seconds per game (20 min)
  max_iterations: null     # turn limit (null = use time_budget only)

prompt_loop:
  enabled: true
  epsilon: 0.02            # accept if score >= best - epsilon
  max_line_changes: 5
  mutator_model: "claude-haiku-4-5-20251001"

scoring:
  survival_weight: 0.30
  population_weight: 0.25
  age_weight: 0.20
  economy_weight: 0.15
  action_success_weight: 0.10
```

### Verification (Phase 0)

Run this to verify everything works:

```bash
python -c "
from gameplay_agent.models import Observations
from gameplay_agent.memory import AgentMemory
from autoresearch.metrics import compute_score
from autoresearch.experiment_log import get_next_experiment_id

# Test game_state field
obs = Observations(game_state='victory')
assert obs.game_state == 'victory'

# Test cumulative metrics
mem = AgentMemory()
mem.create_turn(reasoning='test', actions=[{'type': 'press', 'key': 'h'}],
    observations={'population': '5/10', 'age': 'Feudal Age', 'resources': {'food': 300}})
snapshot = mem.get_metrics_snapshot()
assert snapshot['peak_population'] == 5
assert snapshot['highest_age'] == 'Feudal Age'

# Test scoring
score = compute_score(snapshot)
assert 0 <= score.composite <= 1

print('Phase 0 OK')
"
```

---

## 6. Phase 1: Prompt Optimization Loop

> Status: **NOT STARTED**. This is the next phase to implement.

### Overview

This is the direct autoresearch analog. An LLM proposes changes to the system prompt, a game is played, and the change is accepted or reverted based on the composite score.

### 5.1 Create `autoresearch/prompt_mutator.py`

**Purpose**: Given the current prompt and experiment history, propose a targeted change.

**Implementation details**:

```python
import anthropic
from pathlib import Path

PROMPT_FILE = Path(__file__).parent.parent / "prompts" / "system.md"

# Sections the mutator must NOT modify (output format, game state detection)
PROTECTED_SECTIONS = ["## Output Format", "## Game State Detection"]


class PromptMutator:
    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def read_current_prompt(self) -> str:
        return PROMPT_FILE.read_text()

    def propose_change(
        self,
        current_prompt: str,
        recent_experiments: list[dict],
        failure_modes: list[str],
    ) -> dict:
        """Ask LLM to propose a prompt modification.

        Args:
            current_prompt: Full text of prompts/system.md
            recent_experiments: Last 5 experiments from experiment_log
            failure_modes: Specific failures from most recent game (e.g.,
                "agent got population-capped 3 times",
                "agent never advanced to Feudal Age")

        Returns:
            {
                "description": "Added sheep-gathering priority to Dark Age",
                "old_text": "existing text to replace",
                "new_text": "replacement text",
                "rationale": "why this should improve the score"
            }
        """
        # Build context for the mutator LLM
        experiment_summary = self._format_experiments(recent_experiments)
        failure_summary = "\n".join(f"- {f}" for f in failure_modes) if failure_modes else "None identified"

        system = """You are an expert AoE2 strategist optimizing a system prompt for an AI agent.
Your goal: propose a SMALL, targeted change to the prompt that will improve the agent's game score.

Rules:
- Change at most 5 lines
- Do NOT modify the "## Output Format" or "## Game State Detection" sections
- Focus on strategy, priorities, decision-making heuristics
- Be specific (e.g., "always build 2 houses before advancing" not "build more houses")
- Return JSON with: description, old_text (exact text to replace), new_text (replacement), rationale"""

        user = f"""Current prompt:
```
{current_prompt}
```

Recent experiment results:
{experiment_summary}

Known failure modes from recent games:
{failure_summary}

Propose ONE targeted change to improve the agent's performance."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Parse JSON from response
        # ... (extract JSON from response.content[0].text)

    def apply_change(self, old_text: str, new_text: str) -> bool:
        """Apply the proposed change to prompts/system.md.

        Returns True if the change was applied successfully.
        Validates that old_text exists in the prompt and that
        protected sections are not modified.
        """
        current = self.read_current_prompt()
        if old_text not in current:
            return False

        modified = current.replace(old_text, new_text, 1)

        # Verify protected sections unchanged
        for section in PROTECTED_SECTIONS:
            if section in current:
                # Extract section content and verify it's unchanged
                pass

        PROMPT_FILE.write_text(modified)
        return True

    def revert(self) -> None:
        """Revert prompt to last git-committed version."""
        import subprocess
        subprocess.run(
            ["git", "checkout", "--", str(PROMPT_FILE)],
            cwd=PROMPT_FILE.parent.parent,
        )

    def _format_experiments(self, experiments: list[dict]) -> str:
        lines = []
        for exp in experiments:
            status = "KEPT" if exp.get("accepted") == "true" else "REVERTED"
            lines.append(
                f"  {exp.get('experiment_id')}: score={exp.get('composite_score')} "
                f"[{status}] — {exp.get('change_description')}"
            )
        return "\n".join(lines) or "No previous experiments"
```

**Key design decisions**:
- Uses Haiku (cheap) for mutations, not Sonnet — the mutator doesn't need vision
- Protected sections prevent the mutator from breaking the output format
- `old_text`/`new_text` approach ensures targeted changes (not full rewrites)
- `revert()` uses `git checkout` to undo changes cleanly

### 5.2 Create `autoresearch/orchestrator.py`

**Purpose**: Main loop that coordinates prompt mutation, game running, and accept/reject decisions.

**Implementation details**:

```python
import subprocess
import time
from pathlib import Path

from .experiment_log import (
    get_best_score, get_next_experiment_id, get_recent_experiments, log_experiment
)
from .game_runner import run_game
from .metrics import compute_score
from .prompt_mutator import PromptMutator

REPO_ROOT = Path(__file__).parent.parent
EPSILON = 0.02  # Accept if score >= best - epsilon


class Orchestrator:
    def __init__(self):
        self.mutator = PromptMutator()
        self.best_score = get_best_score(loop="prompt")

    def git_commit(self, message: str) -> str:
        """Commit current changes and return short SHA."""
        subprocess.run(["git", "add", "prompts/system.md"], cwd=REPO_ROOT)
        subprocess.run(["git", "commit", "-m", message], cwd=REPO_ROOT)
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        return result.stdout.strip()

    def git_revert_prompt(self) -> None:
        """Revert prompts/system.md to previous commit."""
        subprocess.run(
            ["git", "checkout", "HEAD~1", "--", "prompts/system.md"],
            cwd=REPO_ROOT,
        )
        subprocess.run(
            ["git", "commit", "-m", "[autoresearch] revert: prompt change rejected"],
            cwd=REPO_ROOT,
        )

    async def run_experiment(self, time_budget: float = 1200) -> dict:
        """Run one full experiment cycle: mutate → play → score → accept/reject.

        Returns dict with experiment_id, score, accepted, description.
        """
        experiment_id = get_next_experiment_id()
        recent = get_recent_experiments(5)

        # 1. Propose a prompt change
        current_prompt = self.mutator.read_current_prompt()
        # Extract failure modes from most recent game's low-scoring components
        failure_modes = self._extract_failure_modes(recent)

        change = self.mutator.propose_change(current_prompt, recent, failure_modes)
        description = change["description"]

        # 2. Apply the change
        success = self.mutator.apply_change(change["old_text"], change["new_text"])
        if not success:
            # Change couldn't be applied (old_text not found)
            return {"experiment_id": experiment_id, "error": "change_not_applicable"}

        # 3. Commit the change
        sha = self.git_commit(f"[autoresearch] {experiment_id}: {description}")

        # 4. Run the game
        result = await run_game(time_budget=time_budget)
        score = result["score"]

        # 5. Accept or reject
        accepted = score.composite >= self.best_score - EPSILON

        if accepted:
            self.best_score = max(self.best_score, score.composite)
        else:
            self.git_revert_prompt()

        # 6. Log result
        log_experiment(
            experiment_id=experiment_id,
            loop="prompt",
            change_description=description,
            score=score,
            accepted=accepted,
            git_sha=sha if accepted else None,
        )

        return {
            "experiment_id": experiment_id,
            "score": score.composite,
            "accepted": accepted,
            "description": description,
        }

    async def run_loop(self, max_experiments: int | None = None, time_budget: float = 1200):
        """Run the autonomous experiment loop.

        Human must start each game manually (Phase 1).
        Orchestrator mutates prompt between games.

        Args:
            max_experiments: Stop after N experiments (None = run forever)
            time_budget: Seconds per game
        """
        count = 0
        while max_experiments is None or count < max_experiments:
            print(f"\n{'='*60}")
            print(f"Experiment {count + 1} — Best score: {self.best_score:.4f}")
            print(f"{'='*60}")

            # Wait for human to start game
            print("Start a new game in AoE2, then press Enter...")
            input()

            result = await self.run_experiment(time_budget=time_budget)

            if "error" in result:
                print(f"Error: {result['error']}")
                continue

            status = "ACCEPTED" if result["accepted"] else "REJECTED"
            print(f"\n{status}: {result['description']}")
            print(f"Score: {result['score']:.4f}")

            count += 1

    def _extract_failure_modes(self, recent: list[dict]) -> list[str]:
        """Identify failure patterns from recent experiments."""
        modes = []
        if not recent:
            return modes

        latest = recent[-1]
        if float(latest.get("population", 0)) < 0.2:
            modes.append("Population stayed very low — agent may not be queueing villagers")
        if float(latest.get("age", 0)) == 0:
            modes.append("Agent never advanced past Dark Age")
        if float(latest.get("economy", 0)) < 0.1:
            modes.append("Very little food gathered — agent may not be assigning villagers to food")
        if float(latest.get("action_success", 0)) < 0.3:
            modes.append("Low action success rate — many actions may be failing")
        return modes
```

**Usage**:

```bash
# Run the orchestrator (human starts each game manually)
python -c "
import asyncio
from autoresearch.orchestrator import Orchestrator
asyncio.run(Orchestrator().run_loop(max_experiments=5, time_budget=1200))
"
```

### 5.3 Git Branching Strategy

All experiments run on a dedicated branch:

```bash
# Before first run
git checkout -b autoresearch/prompt-optimization

# Each experiment:
# 1. mutator writes change to prompts/system.md
# 2. git commit -m "[autoresearch] exp_0001: Added sheep-gathering priority"
# 3. Game plays...
# 4a. If accepted: commit stays, branch advances
# 4b. If rejected: git checkout HEAD~1 -- prompts/system.md + commit revert

# After N successful experiments, merge to main
git checkout main
git merge autoresearch/prompt-optimization
```

### 5.4 Acceptance Criteria (Phase 1)

- [ ] `prompt_mutator.py` can propose, apply, and revert prompt changes
- [ ] `orchestrator.py` runs the full experiment cycle end-to-end
- [ ] After 5 manual experiments, `experiments/results.tsv` has 5 entries with valid scores
- [ ] At least 1 experiment shows an accepted improvement over baseline
- [ ] Git log shows proper commit/revert history

---

## 7. Phase 2: Context Tuning + Strategy Mining

> Status: **NOT STARTED**.

### 6.1 Context Tuning Loop

**Purpose**: A/B test which context configuration produces the best action success rate.

#### Create `autoresearch/context_config.yaml`

```yaml
# Parameters to tune via A/B testing
max_entities: 15              # How many detected entities to pass to LLM
working_memory_turns: 3       # How many recent turns to include
entity_sort_order: "confidence"  # "confidence" | "distance_to_center" | "class_priority"
include_dynamic_context: true # Whether to inject game knowledge DB context
```

#### Create `autoresearch/context_tuner.py`

```python
class ContextTuner:
    """A/B tests context configuration parameters."""

    PARAMETERS = {
        "max_entities": [10, 15, 20, 25],
        "working_memory_turns": [2, 3, 5],
        "entity_sort_order": ["confidence", "distance_to_center", "class_priority"],
    }

    def generate_variant(self, current_config: dict) -> dict:
        """Change one parameter at a time from current config."""
        # Pick a random parameter, pick a random value != current
        ...

    async def run_ab_test(self, config_a: dict, config_b: dict, turns: int = 50) -> dict:
        """Run 50 turns with config_a, then 50 with config_b. Compare action success rate."""
        ...
```

#### Modify `gameplay_agent/game_loop.py` — Read Context Config

In the entity context building section (lines 121-129), make the entity limit configurable:

```python
# Current (hardcoded):
for entity in detected_entities[:15]:

# New (from config):
from autoresearch.context_config import get_context_config
ctx_config = get_context_config()
max_entities = ctx_config.get("max_entities", 15)
sort_order = ctx_config.get("entity_sort_order", "confidence")

# Sort entities based on configured order
if sort_order == "confidence":
    sorted_entities = sorted(detected_entities, key=lambda e: e.confidence, reverse=True)
elif sort_order == "distance_to_center":
    cx, cy = width // 2, height // 2
    sorted_entities = sorted(detected_entities, key=lambda e: abs(e.center[0]-cx) + abs(e.center[1]-cy))
elif sort_order == "class_priority":
    PRIORITY = {"town_center": 0, "villager": 1, "sheep": 2, ...}
    sorted_entities = sorted(detected_entities, key=lambda e: PRIORITY.get(e.class_name, 99))

for entity in sorted_entities[:max_entities]:
    ...
```

Also make working memory depth configurable in `memory.get_context_for_llm()`:

```python
# Current (hardcoded):
recent_turns = list(self.working_memory)[-3:]

# New (from config):
memory_depth = ctx_config.get("working_memory_turns", 3)
recent_turns = list(self.working_memory)[-memory_depth:]
```

### 6.2 Strategy Mining Loop

**Purpose**: Learn which action patterns correlate with good game outcomes, and inject those patterns into the LLM context.

#### Create `gameplay_agent/strategy_db.py`

```python
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "strategy.db"


class StrategyDB:
    """SQLite database for game recordings and mined strategy patterns."""

    def __init__(self):
        self.conn = sqlite3.connect(DB_PATH)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                timestamp TEXT,
                composite_score REAL,
                end_reason TEXT,   -- victory/defeat/timeout
                turn_count INTEGER,
                prompt_sha TEXT    -- which prompt version was used
            );

            CREATE TABLE IF NOT EXISTS turns (
                game_id TEXT,
                turn_number INTEGER,
                timestamp TEXT,
                reasoning TEXT,
                actions TEXT,       -- JSON array
                resources TEXT,     -- JSON dict
                population INTEGER,
                age TEXT,
                game_state TEXT,    -- playing/victory/defeat
                PRIMARY KEY (game_id, turn_number),
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS patterns (
                pattern_id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,  -- human-readable pattern
                condition TEXT,    -- when to apply (e.g., "Dark Age, first 5 minutes")
                action TEXT,       -- what to do (e.g., "queue villagers continuously")
                success_rate REAL, -- win rate when pattern is followed
                sample_count INTEGER,
                confidence TEXT,   -- low/medium/high
                created_at TEXT,
                last_updated TEXT
            );
        """)

    def log_turn(self, game_id: str, turn_number: int, reasoning: str,
                 actions: list, resources: dict, population: int, age: str):
        """Log a single turn's data."""
        import json
        self.conn.execute(
            "INSERT OR REPLACE INTO turns VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, 'playing')",
            (game_id, turn_number, reasoning, json.dumps(actions),
             json.dumps(resources), population, age)
        )
        self.conn.commit()

    def log_game(self, game_id: str, score: float, end_reason: str,
                 turn_count: int, prompt_sha: str):
        """Log a completed game."""
        self.conn.execute(
            "INSERT OR REPLACE INTO games VALUES (?, datetime('now'), ?, ?, ?, ?)",
            (game_id, score, end_reason, turn_count, prompt_sha)
        )
        self.conn.commit()

    def get_proven_patterns(self, min_confidence: str = "medium") -> list[dict]:
        """Get patterns with sufficient confidence for injection into LLM context."""
        conf_order = {"low": 0, "medium": 1, "high": 2}
        min_level = conf_order.get(min_confidence, 1)

        rows = self.conn.execute(
            "SELECT description, condition, action, success_rate, confidence "
            "FROM patterns WHERE sample_count >= 3 ORDER BY success_rate DESC"
        ).fetchall()

        return [
            {"description": r[0], "condition": r[1], "action": r[2],
             "success_rate": r[3], "confidence": r[4]}
            for r in rows
            if conf_order.get(r[4], 0) >= min_level
        ]
```

#### Create `autoresearch/strategy_analyzer.py`

```python
class StrategyAnalyzer:
    """Analyzes game recordings to extract winning strategy patterns."""

    def __init__(self):
        self.db = StrategyDB()
        self.client = anthropic.Anthropic()

    def analyze_recent_games(self, n: int = 3) -> list[dict]:
        """Compare the last N games and extract strategy patterns.

        Sends turn-by-turn data from wins vs losses to an LLM,
        asks it to identify what the winning games did differently.
        """
        # Fetch last N games with their turns
        # Build comparison prompt
        # Ask LLM to identify patterns
        # Store patterns in strategy.db
        ...
```

#### Modify `gameplay_agent/game_loop.py` — Per-Turn Logging

After `memory.create_turn()`, add:

```python
# Log turn to strategy DB (if available)
if strategy_db:
    strategy_db.log_turn(
        game_id=game_id,
        turn_number=iteration,
        reasoning=reasoning,
        actions=actions,
        resources=observations.get("resources", {}),
        population=memory.game_state.population,
        age=memory.game_state.current_age,
    )
```

#### Modify `gameplay_agent/providers/executor_provider.py` — Inject Strategy Patterns

In `_get_dynamic_context()` or a new method, inject proven patterns:

```python
def _get_strategy_context(self) -> str:
    """Inject proven strategy patterns from strategy DB."""
    if not self._strategy_db:
        return ""

    patterns = self._strategy_db.get_proven_patterns(min_confidence="medium")
    if not patterns:
        return ""

    lines = ["## Proven Strategy Patterns"]
    for p in patterns[:5]:  # Limit to top 5
        lines.append(f"- When {p['condition']}: {p['action']} (success rate: {p['success_rate']:.0%})")
    return "\n".join(lines)
```

---

## 8. Phase 3: Automated Game Restart

> Status: **NOT STARTED**.

### 8.1 Problem Statement

Phase 1 requires a human to manually start each game (Step 1 of `orchestrator.run_loop`). This caps experiments at ~5 per day. Phase 3 removes the human bottleneck by scripting the AoE2 game start sequence.

### 8.2 Implementation Plan

**Approach**: AutoHotkey script on the Windows VM that:
1. Detects game-over screen
2. Navigates back to main menu
3. Starts a new Skirmish match with the same settings
4. Waits for "game started" confirmation
5. Hands control back to Python agent

**Files**:
- `vm/restart_game.ahk` — AutoHotkey script (Windows VM only)
- `gameplay_agent/auto_restarter.py` — Python wrapper for VM communication
- Updated `orchestrator.run_loop()` to skip human-input prompt

**Risks**:
- AutoHotkey detection may be unreliable after patches
- New game modes may require additional setup
- May need calibration per game version

### 8.3 Acceptance Criteria

- [ ] Can restart a finished game without human intervention
- [ ] Agent plays 10 consecutive games without manual restart
- [ ] Restart takes <60 seconds from game-over to game-start

---

## 9. Phase 4: Detection Active Learning

> Status: **NOT STARTED**.

### 9.1 Problem Statement

The YOLO detector's accuracy directly limits the agent's gameplay quality. Phase 4 establishes a loop to improve detection:

1. Capture screenshots from real games
2. Identify misclassified or low-confidence entities
3. Hand-label them
4. Retrain YOLO model on the expanded dataset
5. Deploy new model weights
6. Repeat

### 9.2 Implementation Plan

**Files**:
- `detection/active_learning.py` — Identify low-confidence detections, queue for labeling
- `detection/label_studio.py` — Label Studio integration for efficient annotation
- `detection/retrain.py` — Automated retraining pipeline
- `detection/eval.py` — Per-class accuracy evaluation

**Cadence**: Weekly. Detection retraining is expensive (2-6 hours of training + dataset preparation), so we don't want to do it after every game.

**Trigger**: When active-learning queue exceeds 100 unlabeled examples, kick off retraining automatically.

### 9.3 Acceptance Criteria

- [ ] Can identify misclassified detections from gameplay
- [ ] Hand-labeling queue works with Label Studio
- [ ] Retrained model improves per-class accuracy by ≥5%
- [ ] Pipeline runs end-to-end without manual intervention

---

## 10. Phase 5: Training Pipeline Improvements

> Status: **NOT STARTED**.

### 10.1 Problem Statement

Phase 4 is one-shot — retrain once, deploy, hope for improvement. Phase 5 makes it continuous:

1. Phase 4 retraining pipeline runs nightly
2. New model deployed only if it beats current model on validation set
3. Rollback automatic if regression detected in production

### 10.2 Implementation Plan

**Files**:
- `detection/continuous_train.py` — Scheduled retraining
- `detection/model_registry.py` — Track model versions + metrics
- `detection/canary_deploy.py` — Gradual rollout of new models

**Cadence**: Daily on cloud GPU. Validation set held out from training. A/B testing via canary deployment.

### 10.3 Acceptance Criteria

- [ ] Models retrain automatically every night
- [ ] Only models beating validation threshold deploy to production
- [ ] Automatic rollback on regression detection
- [ ] Model registry tracks all versions with metrics

---

## 11. Scoring System

### Composite Game Score

The primary metric for prompt optimization:

```python
composite = 0.30 * survival + 0.25 * population + 0.20 * age + 0.15 * economy + 0.10 * action_success
```

Each component is normalized to [0.0, 1.0]:

| Component | Formula | Cap |
|---|---|---|
| `survival` | `min(time_played / 1200, 1.0)` | 20 minutes |
| `population` | `min(peak_pop / 50, 1.0)` | 50 villagers |
| `age` | `[0, 0.33, 0.66, 1.0][age]` | Dark/Feudal/Castle/Imperial |
| `economy` | `min(total_food / 5000, 1.0)` | 5000 food |
| `action_success` | `success_count / total_actions` | N/A |

### Acceptance Threshold

A change is accepted if:

```python
new_score >= best_score - EPSILON  # EPSILON = 0.02
```

The epsilon prevents rejecting changes that are statistically equivalent.

### Per-Age Score Components

For age-specific analysis (e.g., "did the prompt help Dark Age play?"):

```python
@dataclass
class AgeScore:
    dark_age_score: float    # Performance during Dark Age
    feudal_age_score: float  # Performance during Feudal Age
    castle_age_score: float  # Performance during Castle Age
    imperial_age_score: float # Performance during Imperial Age
```

These break down which age each prompt change most affects.

---

## 12. File Reference

### New Files to Create

```
autoresearch/
  prompt_mutator.py       # Phase 1.1: Propose prompt changes
  orchestrator.py          # Phase 1.2: Run experiment loop
  context_tuner.py         # Phase 2.1: A/B test context configs
  context_config.py        # Phase 2.1: Load context configs
  strategy_analyzer.py     # Phase 2.2: Mine winning patterns from game logs
  auto_restarter.py        # Phase 3: VM game restart

detection/
  active_learning.py       # Phase 4: Identify misclassified detections
  label_studio.py          # Phase 4: Labeling integration
  retrain.py               # Phase 4: Automated retraining
  eval.py                  # Phase 4: Per-class evaluation
  continuous_train.py      # Phase 5: Daily retraining
  model_registry.py        # Phase 5: Track model versions
  canary_deploy.py         # Phase 5: Gradual rollout
```

### Modified Files

```
gameplay_agent/
  memory.py               # Add composite score tracking (Phase 0)
  game_loop.py            # Add game-over detection, time budget, configurable context (Phases 0, 2)
  executor_provider.py    # Inject strategy patterns (Phase 2.2)
  providers/strategist.py # Use proven strategy patterns

autoresearch/
  metrics.py              # GameScore computation (Phase 0)
  experiment_log.py       # Experiment tracking (Phase 0)
  game_runner.py          # Game execution + logging (Phase 0)
  config.yaml             # All phase configs

prompts/
  system.md               # Updated for game_state (Phase 0)
```

---

## 13. Cost Estimates

### Phase 1 Costs (per experiment)

- **Game runtime**: 20 minutes vs Easiest AI = ~$0.50 LLM cost (Claude Sonnet/Opus)
- **Mutator cost**: ~$0.01 per mutation (Haiku)
- **Storage**: Negligible (small TSV files)
- **Per experiment**: ~$0.51
- **Daily (5 experiments)**: ~$2.55

### Phase 2+ Costs

- **Context tuner**: Same as Phase 1
- **Strategy analyzer**: ~$0.10 per analysis (Haiku on game transcripts)
- **Retraining (Phase 4)**: $5-20 cloud GPU per retraining cycle
- **Continuous training (Phase 5)**: $150-600/month for daily cloud GPU

### ROI Considerations

If the agent's average game score improves from 0.45 to 0.65 with prompt optimization:
- That's a 44% improvement in measurable performance
- At 5 experiments/day, expect ~2-3 weeks to find significant improvements
- Total cost: ~$50-100 for Phase 1 experiments

### When to Stop

Stop the autoresearch loop when:
- 10 consecutive experiments show no significant improvement
- Composite score plateaus (delta < 0.01 over 5 experiments)
- Manual inspection of high-scoring games shows prompt is near-optimal

