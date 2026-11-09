# DeepShield project handoff

## Assignment

- Assignment ID: `deepshield-current-transfer-protocol-1`
- Preparation implementation commit:
  `9666d5f880304e5a10f10fb8a9681e0f96c36d38`
- Branch: `prototype/stable-v0.3.1`
- Status: reviewable protocol and CPU-tested dry-run readiness; experiment
  remains unrun and blocked.

## Delivered

- Dated primary-source and official model/pipeline audit through 2026-11-09.
- A differentiated, falsifiable DeepShield-versus-BlurGuard question on a
  held-out FLUX.2 editor at development-matched visual cost.
- Versioned machine-readable protocol:
  `configs/transfer_protocol.2026-11-09.v1.json`.
- Dependency-light validator and compute planner:
  `scripts/validate_transfer_protocol.py`.
- Protocol validation tests in `tests/test_transfer_protocol.py`.
- Detailed methods, rights, metrics, uncertainty, interaction, and limitation
  brief in `docs/CURRENT_TRANSFER_PROTOCOL.md`.

## Evidence boundary

This work adds no images, protected samples, model weights, inference outputs,
GPU timing, provider results, or effectiveness evidence. Every experimental
cell is `unrun`. VETO/FLUX.2 comparisons are a candidate interpretation of
primary sources, not a verified novelty claim or authorization to execute.

## Dry-run result

- 234 planned editor calls.
- 36 planned protection jobs.
- 16.9 provisional GPU-hours for all stages under explicit unmeasured planning
  assumptions.
- 1.267 provisional GPU-hours for the feasibility-only slice.
- Proposed later feasibility ceiling: 2 GPU-hours, $20, or 30 GB of downloads,
  whichever comes first; approval status remains `not-approved`.
- Dollar cost remains `null` unless a live hourly price is supplied for
  arithmetic.

The validator returns `execution_ready: false` with seven blockers:

1. protocol not frozen;
2. rights-cleared assets missing;
3. editor adapter not verified;
4. method implementations not verified;
5. visual budget not frozen;
6. primary metric/reviewer plan not ready;
7. compute not approved.

## Checks

- `python3 -m unittest discover -s tests -v`: 14/14 passed.
- `python3 -m compileall -q deepshield scripts tests`: passed.
- Protocol validator without a price: passed and returned blocked/unrun state.
- Protocol validator with a hypothetical `$2/hour` input: arithmetic-only
  result `$33.80` full / `$2.53` feasibility; no authorization implied.
- `python3 -m json.tool` on the protocol: passed.
- `git diff --check`: passed.
- Gitleaks 8.30.1 staged first-party snapshot: approximately 44.18 KB, zero
  findings.
- Gitleaks 8.30.1 reachable history before this commit: 20 commits,
  approximately 1.73 MB, zero findings.

Redacted scan reports are outside the repository at
`/tmp/deepshield-transfer-protocol-scan.3rQ7L2/`. No secret value was printed.

## Attribution and rights

BlurGuard remains the MIT baseline and ancestor, with its authors credited.
DeepShield remains group work; Ali's contributions are not expanded into sole
authorship. Local baseline checkouts and PDFs were not modified, staged, or
treated as authorized image/data sources.

No image is eligible until it has explicit research/transformation/
redistribution permission and asset-level provenance. Existing repository,
paper, and vendor images are ineligible by default.

## Next review gates

Before even the feasibility slice, review must resolve exact image rights,
FLUX.2 model-card terms, the held-out adapter, the BlurGuard adapter, a fully
loaded/frozen DeepShield surrogate set, primary human-review operations, metric
checkpoint validity, deterministic preprocessing records, measured runtime,
live pricing, and an explicit cap approval.

No model download, inference, GPU, provider/client access, spend, publication,
or protocol execution occurred in this assignment.

## Optional narrow cleanup

The prior release-alignment task left a clean sparse worktree at
`/Users/alihasan/Downloads/DeepShield-main-release`. It contains only the
main-derived README checkout and does not materialize main's large image corpus.
After confirming no further release review needs it, the safe narrow cleanup is
`git worktree remove /Users/alihasan/Downloads/DeepShield-main-release` from the
canonical repository. Do not broadly delete repository assets or untracked
reference directories.
