# Orchestrator Onboarding - EvacOS-MA Repair Campaign

If you are a fresh orchestrator session picking this repo up, read this file first, then read the files listed in Section 1 before taking any action.

---

## 0. Current state (last updated 2026-04-22)

Phase 24 is ACCEPTED, the original repair campaign findings are effectively exhausted, and the post-campaign audit is complete.

Current accepted closeout artifacts:
- Executor summary: `logs/phase24_summary.md`
- Separate screening: `logs/phase24_codex_screening.md`
- Final verdict: `logs/phase24_claude_verdict.md`

Post-campaign audit artifact:
- `logs/phase25_audit.md`

Current health read:
- Phase 23 closed M2/M3.
- Phase 24 disposed of M5, M8, L1, and L2.
- G1, G4, and G5 are closed.
- C5 is refuted.
- The main remaining caveat is the known Windows `tmpdir` / `tmp_path` ACL issue affecting some pytest flows.

Next step:
- Read `logs/phase25_audit.md`.
- It reports no substantive new product-level findings after an audit-time onboarding cleanup.
- Treat the repo as broadly healthy and wait for new user-directed work, while keeping the Windows `tmpdir` / `tmp_path` ACL caveat in mind for future pytest runs.

---

## 1. Required reading (in order)

1. `AGENTS.md` - active repo operating manual
2. `CLAUDE.md` - orchestration spec and process rules
3. `Phase Prompts/README.md` - status table of every archived prompt
4. `logs/review_master_FINAL.md` - original master review that drove the campaign
5. `logs/phase24_claude_verdict.md` - latest accepted phase verdict
6. `logs/phase25_audit.md` - latest whole-project audit, if present

Do not skip these. The repo still carries campaign-specific process assumptions that are not recoverable from code alone.

---

## 2. Campaign state at handoff

- Head base: commit `0236fcb` (post-Phase-10)
- Accepted phases: 11, 11-repair, 12, 12-repair, 12-repair-2, 13, 14, 15, 16, 17, 18, 19, 19-repair, 19-repair-2, 20, 20-repair, 21, 22, 23, 24
- In flight: none
- Post-campaign artifacts: Phase 25 is an audit artifact, not a repair phase

---

## 3. Active role split

Follow `AGENTS.md` for the active role split.

Default setup:
- This interface is the Claude-style orchestrator.
- GLM 5.1 is the default primary executor via `python glm-exec.py`.
- Codex gpt-5.4 is the reviewer / screener / escalation tier via `codex exec`.

Session-specific overrides from live user instruction take precedence over the default split.

Rules:
- Do not use `mcp__codex__*` review tools; use `codex exec` via shell when Codex is the chosen tier.
- Archive every prompt in `Phase Prompts/` before dispatch.
- Append executor-tier changes to `logs/fallback.log`.

---

## 4. Worktree and screening caution

This worktree has long-lived campaign residue and does not use commits as phase boundaries.

Do not rely on:
- `git status --short`
- unscoped `git diff --name-only`
- commit-range diffs like `HEAD~1..HEAD`

For scope checks, use:
- direct reads of the allowed files
- `git diff -- <allowed files>` scoped to the phase surface
- executor/screener artifacts such as `logs/phaseN_summary.md` and `logs/phaseN_codex_screening.md`

This matters because earlier screenings produced false scope alarms when they used the dirty worktree as if it were a clean per-phase branch.

---

## 5. First action on arrival

1. Confirm there is no in-flight repair or undispositioned screening artifact.
2. Read `logs/phase24_claude_verdict.md`.
3. Read `logs/phase25_audit.md` if it exists.
4. Check `Phase Prompts/README.md` for any newer archived prompt that changed the state after this handoff.
5. If no new finding exists, do not restart the old campaign roadmap; wait for new user-directed work.

---

## 6. Non-obvious project details

Things learned the hard way during this campaign:

- Windows paths: prefer forward slashes in shell examples when possible.
- Do not run broad `python -m pytest` across the whole repo unless you have a specific reason; the Windows tempdir cleanup bug can dominate the result.
- `training/rollout.py` intentionally keeps `collect_episode(..., cleanup_env_episode=False)` as the default. Do not flip that default casually.
- `RewardBreakdown` should use typed fields for new reward components instead of relying on loose extras.
- `logs/fallback.log` is the canonical place to record tier changes and quota-driven fallbacks.
- The prompt archive in `Phase Prompts/` is mandatory for reproducibility and auditability.

---

## 7. Historical note

Older sections of this file used to describe Phase 23 as the next step and pointed at Phase 15 as the latest template. Those instructions are obsolete now that Phases 23 and 24 are accepted. If you find other stale campaign references elsewhere, treat them as documentation drift and update them before relying on them.
