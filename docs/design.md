# How it works

The reasoning behind the detector, and the measurements the numbers rest on.
For installing and using it, see the [ReadMe](../ReadMe.md).

Most duplication detectors compare exact structure, so a copy with one extra
statement is invisible, and they report every repeated accessor and guard clause
alongside real problems. Developers learn the output is noise and stop reading
it. This server answers a narrower question: **where is the duplication that is
actually worth someone's afternoon?**

## How it ranks

A sixty-line block copied four times is a far worse problem than a three-line
fragment appearing forty times. The first is a genuine maintenance liability —
every copy is a place a future change has to be repeated and can be forgotten.
The second is almost always a language idiom.

So severity weighs size above frequency:

```
severity = medianLines^1.5 x log2(copies)
```

Frequency enters logarithmically because going from two copies to four matters
far more than going from thirty to forty — by then it is a pattern, not an
incident. Results are also available ordered purely by frequency, since "what is
copied most often" is a different question that also gets asked.

## What it does not report

Repetition that is simply how the language is written gets demoted rather than
ranked:

- **Small and frequent** — at or below `idiom.maxLines` and at or above
  `idiom.minOccurrences`.
- **Spread thinly across the tree** — the same shape once in each of a dozen
  unrelated folders is house style, not one copy-paste incident.
- **Configured exclusions** — path globs and content patterns the team has
  chosen to accept.
- **Contained in a larger finding** — copying a function also copies the loop
  inside it. Only the outermost actionable block is reported, so the same work
  is not counted twice.

Nothing is thrown away: `includeSuppressed` returns these with the reason each
was demoted, so the rules can be checked and tuned.

## Matching

Two passes. Blocks that are textually identical once formatting and comments are
normalised are grouped by hash, which is exact and free. What remains is compared
by meaning using [`jina-embeddings-v2-base-code`](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
embeddings, which is what catches a copy whose variables were renamed.

Block boundaries are inferred without a parser, so every language the team writes
is covered. That means line ranges are approximate — the reported source is
authoritative, not the extents.

## Confidence, and why findings are not simply filtered

Near-miss matching is deliberately inclusive: missing a large repeated block is
worse than offering one that turns out to be a coincidence. So rather than hide
uncertain findings, every one carries a **confidence**, and the reply ships the
scale that explains it:

| Confidence | Meaning |
|---|---|
| `certain` | Identical once formatting and comments are set aside. Not a judgement call. |
| `high` | Almost certainly a copy, typically renamed or lightly edited. Safe to act on. |
| `moderate` | Probably related — read the code first. Shared structure can score here. |
| `low` | Loosely similar. Reported so nothing large is missed; verify before acting. |

Callers that want fewer, safer results pass `minConfidence: "high"`, or raise
`similarityThreshold`. The summary also breaks findings down by confidence, so
the shape of the answer is visible before reading any of it.

### The threshold scales with block size

One fixed similarity number does not work, and the measurements say why. Against
this model:

| | Genuine copy | Unrelated code |
|---|---|---|
| Short blocks (~7 lines) | 0.53 | ≤ 0.22 |
| Long blocks (~100 lines) | 0.89 | up to 0.57 |

Two unrelated hundred-line TypeScript blocks share imports, brace style, naming
habits and control flow, and score highly on all of it. A threshold set for short
blocks therefore groups half the codebase; one set for long blocks misses renamed
functions. So the configured threshold applies to short blocks and rises with
length, reaching 0.8 at a hundred lines. Confidence is judged the same way —
against what a match of *that size* is worth, not against a fixed number.

## Why int8, and why the model is not bundled

The model is downloaded deliberately rather than bundled, because even the
smallest weights are 154 MB — too much to push through an npm install — and
because a few hundred megabytes arriving unannounced mid-question is worse than
being told once that a command needs running.

int8 is the default because it is the only precision CPUs genuinely accelerate.
x86 cores without AVX512-FP16 have no native fp16 compute, so ONNX Runtime
inserts cast nodes per layer and an fp16 model typically runs *slower* than fp32,
while int8 uses VNNI directly. It is also the smallest download. fp32 is offered
for accuracy; fp16 is not offered at all.

**The server runs without the model.** Duplication queries return an empty result
with an explanation of what to install, never an error:

```yaml
status:
 pendingFiles: 3
 modelStatus: not-installed
notice: Duplication analysis needs the jinaai/jina-embeddings-v2-base-code model,
  which is not installed. Run 'duplication-mcp download-model' (or
  'node dist/index.js download-model' from the install directory), ~160 MB, to enable it.
duplications: []
```

## Staying current

Embedding a project takes minutes, which nobody will wait for mid-question. So
changed files are queued and embedded in the background, and questions are
answered from whatever is ready.

The cost of that choice is that an answer can be out of date, so **every reply
reports how many files are still queued**:

```yaml
status:
 pendingFiles: 12
notice: 12 file(s) are queued for embedding, so recent changes may not be
  reflected yet. Ask again shortly for a complete picture.
```

When nothing is queued, both fields are absent and the answer is complete.

Vectors are cached in SQLite keyed by *content*, not location, so moving a block,
re-indenting it or adding a comment all reuse the stored vector, and identical
blocks across twenty files are embedded once. A branch switch is noticed and
re-scanned automatically.

## Where the numbers came from

Every threshold in the code was measured against the real model rather than
chosen by intuition, and the first attempt was wrong in a way only measurement
caught. The initial `similarityThreshold` was 0.92, on the assumption that near
copies score close to 1. They do not: the model puts renamed-but-identical logic
at 0.55, so at 0.92 the semantic half of the analysis would have found nothing
at all while appearing to work.

The calibration scripts that produced the tables above compare small focused
functions and large real blocks separately, which is how the size effect came to
light. `tests/integration/real-model.test.ts` locks the conclusions in: it fails
if a model change ever collapses the gap between genuine copies and unrelated
code.

## Verification

92 unit and end-to-end tests run without the model, using a structure-aware stub
embedder, so the pipeline is exercised offline. 7 integration tests run against
the real model and cover what a stub cannot prove — that renames land above the
threshold, that unrelated code and same-shaped-but-different code land below, and
that the same logic written in another language is still recognised.

Beyond the suites, the detector has been run over a real 7,400-line codebase.
That is what surfaced the size-scaling problem, the window-overlap double
counting, and four separate chunker defects.
