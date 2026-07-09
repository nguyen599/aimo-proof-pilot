# Prime-RL OPD Batching Notes

Date: 2026-07-09

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

## Why This Can Be Inefficient

For the proof OPD workflow, one original problem can expand into many possible
training traces:

```text
bs * (N + N*V + N*V*M) * (R + 1)
```

Example:

```text
bs=2, candidates N=4, verifiers V=4, meta M=2, refine rounds R=2
2 * (4 + 4*4 + 4*4*2) * 3 = 312 possible training samples
```

Waiting for a whole problem/group/refinement tree before training can delay the
first backward pass, especially when long proof generations are uneven.

For OPD, a better behavior may be to train once a useful amount of rollout work is
ready, such as about 64 completed rollout traces, without waiting for the entire
question tree.

## Candidate Future Changes

No code change is made in this note. Possible future changes:

- Set `PRIME_GROUP_SIZE=1` for OPD to avoid waiting for group completion.
- Add an OPD-specific streaming mode in `TrainSink` that finalizes non-group-scored
  rollouts immediately while keeping the existing behavior for GRPO/group-scored
  environments.
- Add a threshold like `prime_opd_stream_batch_rollouts=64`, so the orchestrator
  writes a training batch after roughly 64 completed rollout traces.
- Add token-based batching at the orchestrator level for OPD, so batches target a
  useful packed-token budget rather than a raw rollout count.
- Log these separately:
  - orchestrator pending finalized rollouts
  - orchestrator pending tokens
  - number of samples in the shipped `TrainingBatch`
  - number of packed trainer microbatches
  - per-rank packed token counts

The immediate debugging conclusion is that `batch_size=1` in the trainer log does
not prove `PRIME_BATCH_SIZE` was ignored. It means the shipped training payload
packed into one local microbatch on that trainer rank.
