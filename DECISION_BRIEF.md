# Decision brief: DeepShield P0 security remediation

**Date:** 2026-07-18
**Decision status (updated 2026-07-19):** Ali approved removing former group
collaborators and pursuing detachment from the BlurGuard fork network. No key
rotation or history purge is indicated by the evidence available today.

## 2026-07-19 ownership update

- GitHub removed `sachinjojode`, `bohuali23`, and `ac3295-lab` from
  `AliHasan-786/DeepShield`; all three removals returned HTTP 204.
- A post-change collaborator audit shows only `AliHasan-786`, with admin
  access.
- The public README already gives BlurGuard prominent attribution in its
  opening, baseline comparison, repository structure, and references.
- GitHub accepted the permanent **Leave fork network** confirmation.
- A post-detachment API audit reports `isFork: false` and `parent: null`.

## Recommended call

Accept the secrets-remediation result as complete: the named private key and
local data files were not found in reachable Git history, and content-based
history scans found no leaks. Keep the CaseKick `.env` local and ignored.

The solo-ownership cleanup is complete. DeepShield is a standalone repository
controlled only by Ali, while the README retains prominent BlurGuard
attribution.

## Findings

### DeepShield

- The local `origin` fetch and push URL is
  `https://github.com/AliHasan-786/DeepShield.git`.
- GitHub reports the public repository owner as `AliHasan-786`; the
  authenticated account is `AliHasan-786` / Ali Hasan (user ID `80620634`) and
  has admin access.
- `deepshield-key.pem` is not present in the current working tree, is not
  tracked, and does not appear by path in any fetched branch, tag, reflog
  commit, or unreachable local commit.
- No `.env`, `*.pem`, or `*.db` path appears in the reachable Git object list.
- Gitleaks 8.30.1 found zero leaks across:
  - the bulk scan of all fetched reachable history: 17 patch-bearing
    commits, approximately 1.66 MB;
  - an independent per-commit pass covering all 18 reachable commit objects,
    including both root commits;
  - all reflog history: 19 commits, approximately 1.66 MB;
  - the two otherwise unreachable local commits, scanned individually.
- The fetched remote heads matched the heads advertised by GitHub at audit
  time: `main`, `feature/ensemble-v0.3`, and
  `prototype/stable-v0.3.1`.
- `.gitignore` now explicitly blocks `.env`, `*.pem`, and `*.db`.

**Conclusion:** There is no evidence that `deepshield-key.pem`, an environment
file, a database, or another Gitleaks-detectable secret was committed or pushed
to the currently advertised DeepShield history. Therefore this audit does not
justify rotation or a destructive history rewrite.

### CaseKick

- The audited repository is
  `https://github.com/bohuali23/casekick.git`, a private repository owned by
  the teammate account `bohuali23`. Ali has write, not admin, permission.
- A local `.env` and `dashboard/backend/casekick.db` exist. Both are ignored by
  CaseKick's existing `.gitignore`; neither is tracked.
- The only tracked matching path is `.env.example`, which is the intended
  template.
- No `.env`, `*.pem`, or `*.db` path appears in any fetched branch, tag,
  reflog commit, or unreachable local commit.
- Gitleaks found zero leaks across all 55 reachable commits.
- Gitleaks found one generic API key in the ignored local `.env` at line 17.
  The value was redacted and was not printed or copied. The local database
  produced no finding, although a binary database scan is not proof that every
  stored value is non-sensitive.

**Conclusion:** CaseKick contains a real local credential, but there is no
evidence it was committed or pushed. No teammate notification, force-push, or
rotation is triggered by the Git evidence. The `.env` must remain local and
ignored.

## GitHub ownership and solo-control status

DeepShield is owned by Ali's personal GitHub account, not by a teammate or an
organization. It is now solo-controlled and fully standalone:

- GitHub reports `isFork: false` and no parent repository.
- Ali is now the only collaborator and the only admin.
- The public README still credits BlurGuard prominently.

The collaborator and fork-network settings were changed on 2026-07-19 after
Ali's explicit approval and then independently verified through GitHub's API.

## Evidence and reproducibility

Commands used included:

```bash
git fetch --all --prune
git log --all --full-history --name-status -- '*.pem' '.env' '*.env' '*.db'
git rev-list --objects --all
git reflog --all
git fsck --no-reflogs --unreachable
gitleaks git --redact=100 --log-opts="--all" .
gitleaks git --redact=100 --log-opts="--reflog --all" .
gh repo view AliHasan-786/DeepShield
gh api repos/AliHasan-786/DeepShield/collaborators
```

Redacted scan reports are stored outside both repositories at
`/tmp/deepshield-p0-scan-2026-07-18/`. No secret value is included in this
brief.

## Limits and residual risk

- GitHub cannot prove through the current advertised refs that a secret never
  existed in a branch that was deleted or force-pushed before this audit.
- Gitleaks can miss unsupported secret formats and cannot meaningfully inspect
  every value inside a binary database.
- A local credential can still be exposed outside Git, such as through logs,
  chat, cloud backups, or an accidentally shared archive; those channels were
  outside this repository audit.
- Because no exposed credential was found, rotation remains Ali's decision and
  was not performed.

## Scope confirmation

No history was purged, no force-push occurred, no credential was rotated, no
GitHub access setting changed, and no Tier-3/core DeepShield build work began.

This statement describes the July 18–19 P0 remediation only. A later, separate
checkpoint added an offline evaluation harness and portfolio documentation. It
did not run a GPU experiment, download models, or establish a robustness result.
