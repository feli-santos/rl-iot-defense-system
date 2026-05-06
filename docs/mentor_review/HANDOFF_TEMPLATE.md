<!-- Copy this file to <NN>_HANDOFF.md when closing each step. -->
<!-- Fill every section. Empty sections must say "n/a — <one-line reason>". -->

# Step `<NN>` — `<step name>` — Mentor Review Handoff

**Closed:** `<YYYY-MM-DD HH:MM TZ>`
**Author (agent):** `<agent identifier>`
**Reviewed phase / scope:** `<Phase N or "framing">`
**Status:** `<completed | partial | blocked>`

---

## 1. What was reviewed

### Artifacts
- `<repo-relative path>` — `<one-line role>`
- …

### Code
- `<repo-relative path>` — `<one-line role>`
- …

### Docs
- `<repo-relative path>` — `<one-line role>`
- …

---

## 2. Verdict

`<PASS | PASS-WITH-FIXES | FAIL>`

`<2–4 sentences explaining the verdict in plain English. Lead with the
result. Cite numbers, figure IDs, and gate IDs by name.>`

---

## 3. Findings (priority-ordered)

1. **[severity: blocker | major | minor | nit]** `<one-line headline>`
   `<2–6 sentences. Where (file:line). Why it matters. Recommended fix.>`

2. …

If a finding contradicts an earlier handoff, link the earlier file and
mark this finding as a *correction*. Do **not** edit the earlier file.

---

## 4. Actions taken in this session

- [files added]
- [files edited]
- [files deleted]
- [tests added / changed]
- [scripts added / refactored]
- [results re-run, if any — include git SHAs of any new manifests]

If no actions were taken, write "Read-only review — no source changes."

---

## 5. Outstanding actions for the next session

Each item must be checkable and concrete. Avoid "investigate further."

- [ ] `<exact file or symbol>` — `<exact action>` — `<acceptance criterion>`
- …

---

## 6. How to resume

```bash
# Re-open the project
cd /Users/felipe.santos/Projects/rl-iot-defense-system

# Activate the environment
source .venv/bin/activate

# Verify the project is in the state this handoff claims
git rev-parse HEAD                 # expect: <commit SHA at handoff>
git status                         # expect: clean (or list of expected dirty files)
pytest -q                          # expect: <count> passed
ls docs/mentor_review/             # expect: this file is the highest <NN>_HANDOFF.md
```

If any of those expectations fail, **stop** and surface the divergence
before continuing.

---

## 7. Context-loading recipe for a fresh agent

Read these files **in this order**, in full, before doing any work:

1. `docs/mentor_review/README.md` — directory purpose & conventions
2. `docs/mentor_review/<previous>_HANDOFF.md` — only if applicable
3. `docs/mentor_review/<NN>_<step>.md` — this step's full memo
4. `docs/results/<phase>_<name>/RESULTS.md` — current phase scientific record
5. `docs/results/<phase>_<name>/PLAN.md` — current phase plan (frozen)
6. `<NN>_HANDOFF.md` (this file) — the resume point
7. `<other key files specific to the next step>` — list each with a
   one-line rationale

Skim these for reference if needed (do not read in full):

- `docs/thesis_results_map.md`
- `docs/architecture.md`
- root `README.md`

---

## 8. Open questions for the user

If anything could not be decided unilaterally and needs the candidate's
input, list it here. Otherwise: "n/a".

- `<question>`

---

## 9. Risks introduced or noticed

- `<risk>` — likelihood / impact / mitigation

---

## 10. Sign-off

The next session may proceed when **either**:

- the candidate has acknowledged this handoff (via commit, comment, or
  out-of-band confirmation), **or**
- the "Outstanding actions" list above is empty.
