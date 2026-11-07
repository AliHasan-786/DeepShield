# Manual steps for Ali: first DeepShield GPU pilot

> **PAUSED — not authorized for the 2026-11-07 portfolio checkpoint.** This is
> an archival operational plan, not a current to-do list. Do not provision a
> GPU, accept a paid service, or download models unless a later assignment
> explicitly approves the experiment and its dollar cap.

Do not start this checklist until the local implementation has been committed
and the fake-adapter test suite passes. Codex can prepare and validate commands,
but Ali owns spending, credentials, model-license acceptance, and instance
start/stop actions.

## 1. Make three explicit choices

Reply to the decision brief with:

1. `research artifact` or `startup track` — recommendation: research artifact;
2. the maximum total pilot spend in USD — recommendation: a two-hour ceiling
   with a dollar cap based on the live AWS console price;
3. whether synthetic people are approved for the pilot — recommendation: yes.

Do not use photographs of real people or name a real editing company in a
critical result without a separate approval.

## 2. Put a billing guardrail in AWS

In the AWS console:

1. Open **Billing and Cost Management → Budgets → Create budget**.
2. Create a cost budget at or below the approved pilot cap.
3. Add email alerts before 100% of the cap (for example at 50% and 80%).
4. Check the live on-demand price for `g5.xlarge` in the intended region.
5. Check the EC2 service quota for G-family On-Demand instances.

An AWS Budget is an alert, not an automatic hard shutdown. The two-hour stop
time remains mandatory.

## 3. Handle credentials yourself

- Use an SSH key that Ali controls. Never put the private key inside this repo.
- Enter Hugging Face tokens only on the GPU host with the official login flow.
- Accept any gated model licenses in Ali's Hugging Face account before the run.
- Never paste AWS, SSH, or Hugging Face secrets into Codex or a tracked file.

The SD1.5/SDXL pilot should be attempted before any gated SD3.5 or larger Flux
run.

## 4. Prepare the GPU host

Use Python 3.10 or 3.11 with a working CUDA/PyTorch installation:

```bash
git clone https://github.com/AliHasan-786/DeepShield.git
cd DeepShield
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements_deepshield.txt -r requirements-evaluation.txt
```

After the approved synthetic sample pack and protected variants exist, validate
the manifest without loading a model:

```bash
python scripts/plan_transfer_eval.py configs/evaluation.pilot.json \
  --seconds-per-run 120 \
  --gpu-hourly-usd YOUR_LIVE_AWS_PRICE \
  --json
```

## 5. Run only the draft calibration pilot

```bash
nvidia-smi
time python scripts/run_transfer_eval.py configs/evaluation.pilot.json \
  --allow-draft \
  --output-dir evaluation_outputs/pilot
```

If SDXL runs out of memory, retry once with `--cpu-offload`. Stop rather than
changing the model, seed, JPEG quality, prompt, or thresholds mid-run.

## 6. Stop spending immediately after the pilot

1. Copy the runtime, peak VRAM, errors, and output directory location into the
   decision brief.
2. Stop the EC2 instance in the AWS console. Terminate it if it is disposable.
3. Confirm the EC2 instance state is **Stopped** or **Terminated**.
4. Check for paid attached storage, idle Elastic IPs, and other continuing
   resources.

Do not start the full matrix. Codex first uses the pilot timing and error rate
to calculate a new explicit budget gate and freeze the evaluation protocol.
