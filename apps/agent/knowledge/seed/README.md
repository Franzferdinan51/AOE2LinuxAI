# Knowledge seed

Memories the agent loads into its prompt cache at startup, and rule files the
policy tier evaluates each frame.

- `findings.md`, `prompt_rules.md`, `baseline.tsv` — long-form rules and past
  evidence. The MemoryChain loader reads these from `apps/agent/src/memory_chain.py`.
- `rules/*.yaml` — the seed policy rules. Loaded by
  `apps/agent/src/policy/rules.py`.
- `archive/*.md` — historical memory fragments kept for audit; the loader
  skips them.
