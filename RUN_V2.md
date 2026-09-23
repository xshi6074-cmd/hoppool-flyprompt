# `run_v2.sh` design and usage

`run_v2.sh` is a small FIFO experiment scheduler for one Linux training
server. Its user interface is intentionally limited to:

1. edit the `USER CONFIG` section in `run_v2.sh`;
2. preview with `bash run_v2.sh --dry-run`;
3. launch with `bash run_v2.sh`.

There is no YAML/JSON file and no long terminal command to maintain.

## Configuration model

The three concepts are:

- `GPU_POOL`: physical GPU IDs available to this batch;
- `COMMON_ARGS`: `main.py` arguments shared by every task;
- `add_task`: one named experiment configuration plus its seed list.

Each `task x seed` pair becomes one independent job. For example:

```bash
GPU_POOL=(1 2 4)

COMMON_ARGS=(
    --dataset cifar100
    --data_dir /data/datasets/cifar-100-python
    --n_tasks 5
    --num_epochs 1
)

add_task "hfpool" "1 2 3" \
    --method hfpool \
    --backbone vit_base_patch16_224 \
    --backbone_path /path/to/ViT-B_16.npz \
    --hopfield_trainable o \
    --num_pooling_heads 1

add_task "plain_vit" "1 4" \
    --method baseline \
    --backbone vit_base_patch16_224 \
    --backbone_path /path/to/ViT-B_16.npz
```

This produces five jobs. At most three run simultaneously because the GPU pool
contains three GPUs.

Task arguments are appended after `COMMON_ARGS`. If an ordinary argparse
option is intentionally repeated, the task-local value therefore wins. Do not
put `--gpu` or `--seeds` in either argument list: the scheduler owns both and
rejects them during preflight.

This interface deliberately does not invent an automatic Cartesian-product
sweep. If two settings describe two experiments, write two explicit
`add_task` blocks. That keeps the resolved experiment list readable and makes
accidental job explosions less likely.

## Queue semantics

The scheduler maintains one FIFO queue and one active job slot per GPU:

```text
task/seed expansion -> FIFO queue -> first free GPU -> one main.py process
                                      ^                     |
                                      +---- GPU released ----+
```

GPU IDs are not statically paired with tasks. As soon as a job finishes, that
GPU receives the next pending job. A failed job is recorded and releases its
GPU just like a successful job, so unrelated queued experiments continue. The
scheduler exits nonzero after the whole queue drains if any job failed.

`run_v2.sh` invokes:

```text
main.py --gpu <physical-id> --seeds <one-seed> <common args> <task args>
```

`main.py` reads `--gpu` before importing PyTorch and exposes only that physical
GPU through `CUDA_VISIBLE_DEVICES`. The training code then sees it as
`cuda:0`. Each scheduler job receives exactly one seed, even though
`configuration/config.py` also supports several seeds in one process.
One-process-per-seed is what allows dynamic GPU scheduling and isolated logs.

## Commands

```bash
# Inspect task expansion and the exact commands. Launches nothing.
bash run_v2.sh --dry-run

# Normal use: launch one detached master screen.
bash run_v2.sh

# Attach to the scheduler.
screen -r <name printed by run_v2.sh>

# Debug the scheduler in the current terminal.
bash run_v2.sh --foreground

# Exercise queue refill and artifact writing without Python or GPUs.
bash run_v2.sh --self-test
```

`--dry-run` validates configuration structure, reserved arguments, GPU IDs,
task names, required options and seeds. It intentionally does not require
server-only dataset/checkpoint paths to exist, so it can also run on the local
Windows checkout. A real foreground or detached launch fails before scheduling
if an explicitly configured `--data_dir` or `--backbone_path` is unreadable.

The default launch requires GNU Bash and `screen`. Closing the SSH connection
does not stop the master screen. Detach from an attached screen with
`Ctrl-A`, then `D`.

## Artifacts

Each batch creates:

```text
results/run_v2/<run-id>/
├── run_meta.txt          # commit, GPU pool, counts, start/end state
├── resolved_jobs.tsv     # every expanded command before GPU assignment
├── events.tsv            # job start/finish, GPU, exit code and log path
├── run_v2.snapshot.sh    # exact scheduler/config snapshot
├── logs/
│   └── <job-id>.log      # stdout and stderr for one task/seed
└── status/
    └── <job-id>.status   # internal atomic completion marker
```

The snapshot records uncommitted configuration edits too. `run_meta.txt`
records the current Git commit so the code revision and Bash configuration can
be reconstructed separately.

## Preflight boundaries

The runner verifies scheduling and launch inputs; it does not claim that a
model configuration is semantically valid. In particular, `--dry-run` cannot
prove that an NPZ matches a backbone, that CUDA memory is sufficient, or that a
forward pass succeeds. Use `--foreground` for the first small smoke experiment,
then use the detached mode for the real queue.
