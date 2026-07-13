# Prime-RL OPD Batching Notes

Date: 2026-07-12

This note records the current understanding of why Prime-RL OPD training can show
`Starting forward and backward pass (batch_size=1)` even when the command sets
`PRIME_BATCH_SIZE=2` or higher.

## What the Log Means

The trainer log line:

```text
DEBUG Starting forward and backward pass (batch_size=1)
```

comes from `prime_rl/trainer/rl/train.py`, where `batch_size` is:

```python
batch_size = len(micro_batches)
```

So this is the number of packed microbatches delivered to the local trainer rank.
It is not the orchestrator `batch_size`, not `PRIME_BATCH_SIZE`, and not the
number of original proof tasks.

## Three Different Batch Units

Prime-RL currently has three batch-like units that are easy to confuse:

1. `PRIME_BATCH_SIZE`

   This is the orchestrator/train-sink threshold. In
   `prime_rl/orchestrator/train_sink.py`, it counts finalized `Rollout` objects
   sitting in `pending_batch`. When enough finalized rollouts are ready, the sink
   writes one `TrainingBatch` for the trainer.

   The current long-context command instead sets
   `PRIME_PACKED_SEQUENCES_PER_STEP=64`. The wrapper converts that to:

   ```text
   orchestrator.token_batch_size = max_seq_length * 64
   ```

   Prime then ships a batch after complete environment rollouts have accumulated
   that many prompt-plus-completion tokens. `PRIME_BATCH_SIZE` remains available
   for compatibility when `PRIME_PACKED_SEQUENCES_PER_STEP=0`.

2. `PRIME_GROUP_SIZE`

   This is the rollout group barrier. `TrainSink.add()` first appends an arrived
   rollout to `pending_groups[rollout.group_id]`. It only calls
   `process_group()` when that group reaches `group_size`.

   For OPD, the algorithm scores rollouts independently, so this group barrier is
   probably not algorithmically required in the same way it is for group-relative
   RL algorithms.

3. Trainer microbatches

   After the orchestrator writes a `TrainingBatch`, the trainer-side packer turns
   its samples into packed `[1, seq_len]` microbatches. The train loop logs the
   number of these packed microbatches.

## Packer Behavior

There are two packer paths:

- `SinglePacker`: used when there is one run. It waits for one training batch
  file, then calls `prepare_batch(...)`. It does not explicitly wait until it has
  `seq_len * dp_world_size` tokens before packing.

- `MultiPacker`: used when there are multiple runs. It buffers samples and waits
  until it has roughly:

  ```text
  seq_len * dp_world_size
  ```

  tokens before selecting samples for the next step, with a short timeout escape
  if it has some tokens but not enough.

In both cases, `prepare_batch(...)` packs samples into microbatch bins up to
`seq_len`, then distributes bins across DP workers. One trainer rank can still
see `len(micro_batches) == 1` if the packed batch for that rank has only one bin.

## Current Proof-OPD Boundary

The 8-node hybrid command uses `PRIME_GROUP_SIZE=1` and disables the Proof-OPD
candidate gate. Every multi-turn row generates a proof, then independently takes
the continuation path with probability `0.25`. The other `0.75` stop with a
proof-only trace. Continuing rows run four verifiers, optional meta-verification,
and one refinement round. The hybrid run does not add a generated selector
because pre-rendered selector rows are in the same dataset.

Rows marked `info.execution_mode=single_turn` stop after their one pre-rendered
model call. They use the same multi-turn environment class, so one train
environment can consume both data sources.

There is deliberately no cross-rollout candidate barrier. Each rollout is an
atomic orchestrator unit, so a short proof-only sample can reach OPD token packing
without waiting for unrelated verifier/refinement paths.

The global token targets are:

```text
8,192 * 64 = 524,288 tokens per optimizer step (short backward smoke)
81,920 * 64 = 5,242,880 tokens per optimizer step (long-context target)
```

These are global pre-packing targets. Prime's trainer packer still decides the
actual number of fixed-length microbatch bins and distributes them over data
parallel ranks. The final completed rollout can make a token batch overshoot the
target.

## Candidate Future Changes

Possible future changes, only if full-environment latency becomes a measured
bottleneck:

- Split the environment into durable stage jobs backed by persistent storage.
- Preserve parent/candidate identity without patching the generic verifiers bridge.
- Resume failed stage jobs idempotently and garbage-collect completed candidate trees.
- Log these separately:
  - orchestrator pending finalized rollouts
  - orchestrator pending tokens
  - number of samples in the shipped `TrainingBatch`
  - number of packed trainer microbatches
  - per-rank packed token counts

The immediate debugging conclusion remains that `batch_size=1` in the trainer log
does not prove the orchestrator threshold was ignored. It means the shipped
training payload packed into one local microbatch on that trainer rank.
