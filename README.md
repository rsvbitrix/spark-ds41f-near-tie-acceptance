# Near-tie draft acceptance on spark-ds41f 1107: a single-stream experiment

A site experiment on a switchless TP4 ring of four DGX Sparks running
[spark-ds41f](https://github.com/christopherowen/spark-ds41f) 1107 with the official checkpoint, 2026-11-15.

**This is a lossy speedup.** It changes the output distribution at temperature 1. It is opt-in per request; every
other request takes the stock arithmetic. Nothing here has passed spark-ds41f's determinism gate or release tests.

## Summary

- Single-stream decode on a Spark ring is bound by memory reads. On our ring a step costs about
  `16 ms + 3 ms x (rows verified)`, so the only lever left at one stream is tokens accepted per step.
- At temperature 1 the exact rule rejects the greedy draft token with probability `1 - p` even when that token is
  the target's own best. Reasoning text is full of such near-ties.
- Rule tested: accept the draft token outright when the target gives it at least `k x` the probability of its best
  token; otherwise verify exactly as now.
- At `k = 0.5`, one stream: **reasoning +33% to +69%** (80 → 108, 106 → 142 and 73 → 124 tok/s; the last pair is
  noisy), code +10% to +14%, prose +4% to +12%. The whole set runs at 121.9 tok/s against 95.3 (+28%).
- Accuracy on our graded suites: 272 of 300 against 268 of 300 for exact.
- Cost: with thinking off, hard tasks grow a tail of very long answers (3 of 72 ran to the 65,536-token limit;
  none in exact mode). With thinking on, length and accuracy are unchanged.

## Setup

| | |
|---|---|
| Hardware | 4x DGX Spark, switchless ring, TP4 |
| Weights | `deepseek-ai/DeepSeek-V4.1-Flash-1017`, unchanged |
| Recipe | spark-ds41f-1107, checkout `cc20ed18`, `--max-num-seqs 32`, 1M context |
| Image | 1107 plus two site layers that do not touch speculative decoding: a `/v1/messages` parameter shim, and a reasoning-loop detector that waits for 512 tokens of loop before cutting |

## Why only acceptance is left

Production config, no patch, exact mode, one stream, two runs per prompt (`results/speed/unpatched.json`):

| Prompt | Step, ms | Rows per step | Tokens per step | tok/s |
|---|---:|---:|---:|---:|
| Prose, Russian | 26.8 | 3.75 | 2.42 | 90.1 |
| Prose, English | 27.6 | 3.69 | 2.18 | 78.8 |
| Reasoning | 27.5 | 3.66 | 2.17 | 78.9 |
| Code | 33.5 | 5.79 | 4.58 | 136.9 |
| Tetris in one HTML file | 32.8 | 5.77 | 4.72 | 144.0 |

A row is the current token or one draft token the target verifies. Across prompts the step fits
`16 ms + 3 ms x rows`. With a five-token DSpark block the ceiling is 6 tokens in a 34 ms step, about 176 tok/s,
reached only where the draft is almost never wrong.

Open question, not investigated: the recipe's own grid reports about 16.4 ms for a six-row step, and we see
33-34 ms in a live stream. Our guess is that grid rows share experts; we did not check the grid's code.

## The rule

With a greedy one-hot draft, exact verification accepts draft token `x` with probability `p(x)`.

Typical rule, level `k`:

- accept `x` outright when `p(x) >= k x max_v p(v)`;
- otherwise verify exactly, with probability `p(x)`; a rejected row is resampled as before.

At `k = 0.9` this accepts, in effect, only the target's own best token. At `k = 0.5` it also accepts a token the
target rates at least half as likely as its best. A clear draft miss gets no leniency.

The output moves toward greedy: where the draft is the target's best token or a near-tie, nothing is sampled.

## The patch

[patch/lenient.py](patch/lenient.py) edits one vLLM file,
`vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py`: one inlined device function and one call site, about
40 lines. The Triton path is not touched.

| Environment | Effect |
|---|---|
| `SPARK_NIGHT_LENIENCE_BY_TEMP=1` | the request names its level through its temperature: `1 - k/100` (0.995 is `k = 0.5`) |
| `SPARK_NIGHT_TYPICAL=k` | typical rule at level `k` for every sampled request |
| `SPARK_NIGHT_LENIENCE=l` | plain lenience, accept with `min(1, p / l)` (Leviathan et al., arXiv 2211.17192), for every sampled request |

With none set, the function returns the log probability unchanged. With `BY_TEMP=1`, a request at any
temperature outside `(0.9905, 0.9995)` and `(1.0005, 1.0095)`, including 0 and 1, also gets it unchanged.

Temperature as the switch is a side channel: it was the one per-request scalar already inside the kernel, and it
kept the change to one file. Sampling then runs 0.5% colder than 1 at `k = 0.5`. A real version would take a
sampling parameter.

Plain lenience was the first thing we tried and we do not recommend it. It raises acceptance for clear misses
too: at `l = 0.1` a draft token with `p = 0.1` is always accepted, even when the target is 85% sure of another.

Every run below used a file mount over the image and `SPARK_NIGHT_LENIENCE_BY_TEMP=1`, so exact and fast requests
shared one engine.

## Speed

One stream, idle engine, two runs per prompt, both arms on the same boot (`results/speed/exact.json`,
`results/speed/k05.json`). "First draft" is accepted tokens at draft position 1 per decode step.

| Prompt | Exact, tok/s | k = 0.5 | Change | Tokens per step | Rows per step | Step, ms | First draft |
|---|---:|---:|---:|---|---|---|---|
| Prose, Russian | 88.4 | 91.6 | +4% | 2.38 → 2.53 | 3.67 → 4.13 | 27.0 → 27.6 | 0.70 → 0.73 |
| Prose, English | 77.8 | 86.9 | +12% | 2.15 → 2.49 | 3.63 → 4.10 | 27.6 → 28.6 | 0.64 → 0.72 |
| Code | 135.5 | 149.7 | +10% | 4.59 → 5.11 | 5.72 → 5.93 | 33.9 → 34.1 | 0.90 → 0.96 |
| Tetris | 140.1 | 160.1 | +14% | 4.62 → 5.40 | 5.74 → 6.10 | 32.9 → 33.7 | 0.89 → 0.97 |
| Reasoning, Russian | 79.9 | 108.4 | +36% | 2.22 → 3.29 | 3.68 → 4.97 | 27.8 → 30.4 | 0.59 → 0.85 |
| Reasoning, English | 106.2 | 141.6 | +33% | 3.07 → 4.40 | 4.52 → 5.68 | 28.9 → 31.1 | 0.73 → 0.90 |
| Code review with reasoning | 73.3 | 123.7 | +69% | 1.99 → 4.05 | 3.96 → 5.68 | 27.1 → 32.7 | 0.64 → 0.89 |
| Whole set | 95.3 | 121.9 | +28% | | | | |

- "Whole set" is total output tokens over total decode time, as `nb.py` reports it.
- The two runs of each arm:

  | Prompt | Exact | k = 0.5 |
  |---|---|---|
  | Prose, Russian | 88.3, 88.5 | 91.0, 92.3 |
  | Prose, English | 78.3, 77.3 | 83.8, 90.6 |
  | Code | 132.2, 139.1 | 151.5, 148.0 |
  | Tetris | 141.2, 139.1 | 163.3, 157.1 |
  | Reasoning, Russian | 79.8, 80.0 | 108.0, 108.8 |
  | Reasoning, English | 102.6, 110.1 | 146.3, 137.1 |
  | Code review with reasoning | 85.2, 64.3 | 132.2, 116.3 |

- The code-review pair is noisy in exact mode, so read +69% as a range: the slower fast run against the faster
  exact run is +37%.
- The step grows with acceptance because more rows are verified.
- Prose barely moves: there most of the draft's misses are real ones, not near-ties.
- Exact requests on the patched boot run at the unpatched speed. Against `unpatched.json`, taken on another boot
  six hours earlier, the five shared prompts differ by -2.7% to +1.3%.
- After a later restart with the same patch, a quick check gave reasoning 82.4 → 108.8 tok/s and code
  144.2 → 148.9. That check is not in `results/`.

## Quality

Our own auto-graded suites, not spark-ds41f's. Prompts are not in this package; per-task verdicts, finish
reasons and token counts are (`results/quality/`).

- **fresh**, thinking off, three repeats: 16 code tasks, 20 maths, 8 tool-use agent, 8 instruction-following.
- **deep**: 72 puzzle tasks (eight families, three levels), run once with thinking off and once with thinking
  `low`.

Both modes ran at once on one boot, straight to vLLM on `:8000`, at most 24 of 32 slots busy; 300 tasks per mode.

| Suite (tasks) | Exact | k = 0.5 |
|---|---:|---:|
| Code (48) | 44 | 45 |
| Maths (60) | 58 | 58 |
| Agent with tools (24) | 22 | 22 |
| Instruction following (24) | 16 | 18 |
| Deep, thinking off (72) | 56 | 57 |
| Deep, thinking on (72) | 72 | 72 |
| **Total (300)** | **268** | **272** |

Paired by task and repeat: 13 solved only in exact mode, 17 only at `k = 0.5`. That is one run per arm, so a
difference of four tasks says only that we see no loss of accuracy at this sample size.

Answer length and time on the deep suite:

| | Exact | k = 0.5 |
|---|---:|---:|
| Thinking off: median answer, tokens | 2,062 | 1,962 |
| Thinking off: mean answer, tokens | 2,246 | 5,002 |
| Thinking off: answers over 8,000 tokens | 0 | 5 |
| Thinking off: answers that hit the 65,536-token limit | 0 | 3 |
| Thinking off: cut by the repetition detector | 0 | 1 |
| Thinking off: median / mean time per task, s | 122 / 132 | 84 / 144 |
| Thinking on: median answer, tokens | 2,716 | 2,437 |
| Thinking on: mean answer, tokens | 5,355 | 5,431 |
| Thinking on: answers over 8,000 tokens | 14 | 13 |
| Thinking on: median / mean time per task, s | 121 / 245 | 100 / 222 |

Times are under the shared load above, so compare the columns, not the absolute values.

## The cost

- With thinking on, we see no side effect: same accuracy, same length, and tasks finish sooner.
- With thinking off, the typical hard task also finishes sooner, but a few answers run into a repetition: 3 of 72
  to the token limit and one more cut by the detector. The mean time per task ends up higher than in exact mode.
- We use it only where thinking is on.

## Not covered here

- spark-ds41f's determinism gate, release tests and servebench suites on the patched image.
- A second full boot per arm.
- Any level other than `k = 0.5`.
- Throughput at 8 to 16 streams.
- Quality at temperature 0.
- A long agent session: hours, dozens of tool calls.
- TP3, TP2 and switched shapes.

## Files

| Path | What |
|---|---|
| `patch/lenient.py` | the patcher; fails if the image's source differs from the one it was written against |
| `patch/unit-lenient.py` | prints the device function the patched file builds; no GPU needed |
| `tools/nb.py` | one-stream bench: tok/s, step time, tokens and rows per step, acceptance by position, from vLLM's counters |
| `tools/curve.sh` | `nb.py` over a list of levels on one boot |
| `results/speed/unpatched.json` | production before the patch, five prompts |
| `results/speed/exact.json`, `k05.json` | the patched boot: temperature 1 and temperature 0.995, seven prompts |
| `results/quality/exact/`, `k05/` | per-task verdicts: `fresh.json`, `deep-none.jsonl` (thinking off), `deep-low.jsonl` (thinking on) |

## Reproduce

```bash
# on a node, with the 1107 container running
mkdir -p patch-out
docker cp spark-ds41f:/opt/spark-ds41f/candidate/vllm/vllm/v1/worker/gpu/spec_decode/tilelang_rejection.py patch-out/
python3 -I patch/lenient.py patch-out
# mount patch-out/tilelang_rejection.py over the image's file on every node,
# set SPARK_NIGHT_LENIENCE_BY_TEMP=1 in the container environment, restart the model
python3 tools/nb.py exact --sets text_en,code,think,think_en --extra '{"temperature": 1}'
python3 tools/nb.py k05   --sets text_en,code,think,think_en --extra '{"temperature": 0.995}'
```

## License

Apache-2.0, the same as spark-ds41f and vLLM. See [LICENSE](LICENSE).
