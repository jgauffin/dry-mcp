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
  is not counted twice. A finding that lies almost entirely over a larger one
  (80% of every occurrence) counts as contained too, since two nested blocks a
  line apart otherwise report the same copy twice.
- **Pieces of one block** — a block too long to compare whole is cut into
  overlapping windows, and two windows of one block are never copies of each
  other, however far apart. Written by the same hand in the same style, they
  otherwise score as alike as a genuine copy.

Nothing is thrown away: `includeSuppressed` returns these with the reason each
was demoted, so the rules can be checked and tuned.

## Matching

Two passes. Blocks that are textually identical once formatting and comments are
normalised are grouped by hash, which is exact and free. What remains is compared
by meaning using [`jina-embeddings-v2-base-code`](https://huggingface.co/jinaai/jina-embeddings-v2-base-code)
embeddings, which is what catches a copy whose variables were renamed.

Embeddings score how alike code *reads*, which on a real project is not the same
as whether it was copied: three unrelated test suites sharing describe/it/expect
scaffolding, or a dozen tool classes each with an async `execute`, score as
highly as a genuine copy. So a near-miss also has to keep its **lines**. Each
pair the embeddings accept is aligned line by line, in order, and the share of
the shorter block that lines up is its `alignment`. Two lines are the same line
when they share most of their words, or when every word on them is used at the
same distance from its previous use — which is exactly what survives renaming a
variable throughout, and what code merely shaped alike does not have. Words in
strings count, so two `it('…')` lines naming different tests do not match.

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

### Confidence is capped by alignment

Clearing the size-scaled threshold already puts a block a margin above the
unrelated-code baseline, so judged by embeddings alone nearly every near-miss
came out `high` — on one project all 274 of them, including pairs at 0.53. A
near-miss is therefore rated no higher than its alignment allows: below 0.65 it
is not reported, from 0.7 it can be `moderate`, from 0.75 `high`.

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

The cost of that choice is that an answer can be out of date, so **every
incomplete reply reports how far the index has got**:

```yaml
progress:
 filesInScope: 3510
 filesEmbedded: 1204
 pendingFiles: 2306
 percentComplete: 34
```

When nothing is queued, the block is absent and the answer is complete.

Below half embedded, a project larger than a couple of hundred files gets that
progress **instead of** findings. A ranking drawn from a third of a codebase is
not an early version of the real ranking: the worst duplication is most likely in
the part not yet read, while the reply reads like an answer and invites acting on
whatever was indexed first. Small projects are always ranked as they stand, since
they finish before anyone could ask twice.

## Scope

Without a `duplication.config.json` every source file under the root is
analysed, which on a first run is rarely what was meant. So when the project is
large and nothing has been narrowed, the reply names the file count and the
largest folders and says what to write — the calling agent can edit that file
itself, and the rules are re-read before the next question.

Directories that are repositories of their own — git submodules, vendored clones
— are left out by default. They are another project's code: duplication inside
one cannot be fixed from here, and on a first run in a repository with
submodules it is most of what would be reported. The skipped paths are named in
the reply rather than dropped quietly, because a team that does own its
submodules has to be able to see why their code never appears.

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

The alignment floor was measured on a 393-file TypeScript project whose fully
indexed report had been read by hand. The pairs that were wrongly grouped
aligned at 0.21 at most, and test cases sharing only a fixture's vocabulary at
0.5 to 0.61. Real copies aligned from 0.67 up: repeated test setup at 0.67, a
fixture builder copied into four files at 0.72, adapted functions at 0.79. The
integration suite's fully renamed function, and the same logic rewritten in C#,
both align at 0.8. With the floor at 0.65 that report went from 274 clusters,
all `high`, and 29,563 removable lines to 297 clusters (5 certain, 72 high,
133 moderate, 87 low) and 5,058 lines, with the window artefacts and shape-only
groups gone.

## Verification

The unit and end-to-end suites run without the model, using a structure-aware
stub embedder, so the pipeline is exercised offline. The integration suite runs
against the real model and covers what a stub cannot prove — that renames land
above the threshold, that unrelated code and same-shaped-but-different code land
below, and that the same logic written in another language is still recognised.

Beyond the suites, the detector has been run over a real 7,400-line codebase.
That is what surfaced the size-scaling problem, the window-overlap double
counting, and four separate chunker defects.
