# Third-party boundaries

DeepShield is derived from [BlurGuard](https://github.com/jsu-kim/BlurGuard).
The repository keeps BlurGuard attribution and its MIT copyright notice in the
root `LICENSE`. The Git history also preserves contributions from the original
authors and former DeepShield collaborators.

The following local directories are separate upstream Git checkouts used only
to inspect possible comparison baselines. They are intentionally untracked and
are not distributed as part of DeepShield:

| Local checkout | Upstream | Observed license file | Pinned local commit |
|---|---|---|---|
| `Anti-DreamBooth/` | `VinAIResearch/Anti-DreamBooth` | AGPL-3.0 | `1b0e3634b5883cddcdb44960b3be7fa23ec58b67` |
| `photoguard/` | `MadryLab/photoguard` | MIT | `686bea75c786cb46c88fc396a0cd0ee3d7d28c2e` |
| `mist-v2/` | `psyker-team/mist-v2` | Apache-2.0 | `ef8001ae064642011c18743bc26245bd587e83e4` |
| `MMA-Diffusion/` | `cure-lab/MMA-Diffusion` | CC BY-NC-ND 3.0 | `d5701a5dd1ad4790249b31cbe48ebd6459029e63` |
| `DiffAdvPerturbationBench/` | `vkeilo/DiffAdvPerturbationBench` | no root license file observed | `61f3c77ce4f8950584b6161b1ffa07094b6cde11` |

No code, result, or output from these checkouts is counted as Ali Hasan's
original work. Before any adapter code or public comparison is copied into this
repository, its exact license and transitive assets must be reviewed. In
particular, the absence of an observed root license is not permission to copy,
and the MMA-Diffusion license places noncommercial/no-derivatives restrictions
on the licensed work.

The two locally stored papers are also untracked research references. Their
presence is not a license to redistribute them.
