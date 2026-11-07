# Security and pre-publish checks

DeepShield must not commit credentials, private keys, local environment files,
or local databases. The repository ignore rules cover `.env`, `*.pem`, and
`*.db`, but ignore rules are only a guardrail; they do not replace a history
scan.

Before every push that will make new work public:

1. Confirm the staged paths do not include secrets or local state:

   ```bash
   git diff --cached --name-only
   ```

2. Scan every reachable local and remote commit with Gitleaks:

   ```bash
   git fetch --all --prune
   gitleaks git --redact=100 --log-opts="--all" .
   ```

3. Do not push if Gitleaks reports a finding. Determine whether it is a false
   positive without printing the secret. If a real credential was ever pushed,
   Ali must rotate it before any history-rewrite decision.

4. Confirm GitHub still reports `AliHasan-786/DeepShield` as owned by
   `AliHasan-786` before publishing.

History purges, force-pushes, credential rotation, collaborator removal, and
fork-network changes require an explicit decision from Ali.
