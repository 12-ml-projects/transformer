# Transformer

A GPT-1-style decoder-only transformer built from scratch in PyTorch, benchmarked
against a language whose **best possible loss is known exactly**.

When you train a language model on real text, you never know how good "good" is:
a validation loss of 1.9 might be excellent or terrible. This project avoids that
by first training on an artificial language generated from a probabilistic
grammar. For such a language the entropy per token can be computed in closed
form, which gives a hard floor for the cross-entropy loss. The model is correct
when its validation loss approaches that floor and never beats it.

## The artificial language

The language is a probabilistic context-free grammar (PCFG) with number agreement
and nested relative clauses:

```
S       -> NP_sg VP_sg Dot [0.5] | NP_pl VP_pl Dot [0.5]
NP_sg   -> Det_sg Nom_sg [0.8] | Det_sg Nom_sg Rel [0.2]
Rel     -> "that" NP_sg Vt_sg [0.5] | "that" NP_pl Vt_pl [0.5]
Nom_sg  -> Adj Nom_sg [0.3] | N_sg [0.7]
...
```

Sampled text looks like this:

```
the cat sees . a cat that this cat chases sees some dogs .
the dog sees a dog that some small dogs that those dogs that these small cats see chase see .
```

Getting the verbs right in the second sentence requires tracking agreement across
arbitrarily deep embeddings, so a model can only reach the floor if it has
genuinely learned the grammar.

The grammar lives in a plain text file and is loaded with NLTK. Every word is a
token, so no tokenizer is needed. A torch `Dataset` samples sentences, joins
them into a continuous stream and cuts it into fixed-length windows.
Sampling is deterministic per index and seed, so train, validation and test sets
are independent by construction.

## The entropy floor

For a nonterminal `A` with rules `r`, the entropy of everything `A` generates
satisfies

```
H(A) = −Σ_r p(r) log p(r)  +  Σ_r p(r) Σ_{B ∈ rhs(r)} H(B)
```

and the expected length `L(A)` has the same form. Both are linear systems,
`(I − M) x = c`, where `M[A, B]` is the expected number of `B`s produced by one
expansion of `A`. Solving them gives the per-token entropy `H(S) / L(S)`.
The solver also checks the grammar's consistency (spectral radius of `M` < 1).

The result is cross-checked by an independent Monte Carlo estimator. The sampler
accumulates each sentence's log-probability while it samples, and the tests
confirm that it matches NLTK's Viterbi parse probability.

| grammar | analytic | Monte Carlo (80k sentences) |
|---|---|---|
| main language | **1.4046 nats/token** | 1.4051 |
| coin flip `S → a \| b` | ln 2 | ln 2 (exact) |

This equals the entropy of the text only if the grammar is unambiguous, and a
test parses sampled sentences to verify that each has exactly one parse.

One subtlety: the first token of each training window is only ever an input,
never a target, so the achievable loss per window is slightly *below*
`H(S) / L(S)`. The tests account for this correction.

## The model

A decoder-only transformer following GPT-1, written from the individual pieces:

- token embedding plus sinusoidal positional encoding, stored as a buffer;
- masked multi-head self-attention with a causal mask;
- a position-wise feed-forward network with GELU;
- post-LayerNorm residual blocks;
- a linear output layer producing logits for `CrossEntropyLoss`.

The training loop is a generator that yields `(train_loss, valid_loss)` after
every epoch. The caller decides when to stop, log or plot.

## What the tests certify

Beyond unit-level checks on shapes, determinism and batch independence, the model
is tested end to end against languages with known floors:

| test | language | what it proves |
|---|---|---|
| `test_can_reach_0_loss_if_possible` | `test1 test2 test3 test4 …` (entropy 0) | the model and training loop can learn: gradients flow and the output is wired correctly |
| `test_can_reach_theoretical_best_loss` | `a x b \| c x d` | the validation loss approaches the analytic floor and never goes below it |

The second test is the important one. Predicting `b` versus `d` after `x`
requires attending two positions back, so the test **fails when attention is
disabled**. The "never below the floor" check catches information leaks such as a
broken causal mask. Validation uses a separately seeded dataset, so the model
cannot pass by memorising.

The estimator tests pin the analytic solver against grammars with exact answers,
and compare it with the Monte Carlo estimate on the main language.

## Experiment: how close does the model get?

The tests show that the model *can* reach the floor on small languages. The
experiment in [`experiments/simple_language`](experiments/simple_language/main.py)
measures how close it gets on the main language, and with which architecture.

### Setup

- **Search:** [Optuna](https://optuna.org/) with a TPE sampler over the number of
  layers, model width, heads, dropout and learning rate.
- **Pruning:** Hyperband stops unpromising trials early. It can be switched to a
  median pruner or disabled with `--pruner`.
- **Tracking:** every trial is an [MLflow](https://mlflow.org/) run nested under
  one parent run per study. Each run logs its hyperparameters and its per-epoch
  train and validation loss. The best trial is tagged `best=true`, and the parent
  run records its id.
- **Data:** train, validation and test sets are sampled from the grammar with
  different seeds, so they are independent. The best model is checkpointed during
  the search and evaluated **once** on the test set, which played no part in
  selecting it.

### Result

The best of 70 trials (48 of them pruned), trained for 25 epochs:

| | |
|---|---|
| architecture | 4 layers, width 64, 2 heads |
| learning rate / dropout | 2.6e-3 / 0.10 |
| validation loss | 1.4327 |
| **test loss** | **1.4368 nats/token** |
| entropy floor | 1.4046 nats/token |
| gap | 0.032 nats/token (perplexity 4.21 vs 4.07) |

![Train and validation loss of the best trial over 25 epochs](plots/train_vaild_loss_lang_1.svg)

The test loss is within about 2% of the theoretical optimum. Measured against the
per-window floor, which is lower (roughly 1.39, see above), the gap is about
0.04 nats/token. The test loss is only slightly above the validation loss, which
shows that selecting the best of 70 trials on the validation set introduced
little optimism.

Optuna's hyperparameter importance ranks the **learning rate** and **dropout** as
the most important parameters, ahead of the architecture itself.

### Running it

```bash
# 1. start an MLflow tracking server (UI at http://localhost:8888)
poetry run mlflow server --port 8888

# 2. run the study: trials, epochs and pruner are configurable
poetry run python -m experiments.simple_language.main \
    --language-spec tests/language_specs/language_1.txt \
    --n-trials 50 --n-epochs 10 --pruner hyperband

# 3. optional: Optuna's dashboard for hyperparameter importance and slice plots
poetry run optuna-dashboard sqlite:///experiments/simple_language/optuna.db \
    --port 8081 --server wsgiref
```

The script prints the best trial and its test loss when it finishes. The Optuna
study is stored in `experiments/simple_language/optuna.db` and resumes where it
left off when you run it again.

The dashboard needs `--server wsgiref`. With the default server it runs on
gunicorn, whose forked workers inherit the SQLite connection, which then fails
with `disk I/O error`.

## Layout

```
src/
  data/         grammar loading, sampling, vocabulary, windowed Dataset, collate
  estimators/   analytic (linear system) and Monte Carlo entropy estimators
  model/        embedding, attention, feed-forward, block, decoder
  experiment/   training loop
experiments/
  simple_language/  hyperparameter search on the artificial language
tests/
  language_specs/   the grammars used in tests
  integration/      data, estimator and model tests
```

## Getting started

Requires Python 3.13 and [Poetry](https://python-poetry.org/).

```bash
poetry install
poetry run pytest                  # full test suite, ~10 s on a laptop CPU
tox                                # tests, flake8, black/isort, mypy
```

## Roadmap

1. **Artificial language experiments.** The hyperparameter search is done (see
   above). Still to do: generate sentences from the trained model and check with
   the grammar how many of them are grammatical, and which errors it makes.
2. **Tokenizer.** Write a byte-level BPE tokenizer from scratch, with streaming
   training on word counts, and encode corpora to memory-mapped token files.
3. **Modernised training.** Add pre-LayerNorm blocks (GPT-2 style) behind a flag,
   plus a cosine learning-rate schedule with warmup.
4. **Scaling laws.** Use Chinchilla's approach 2 (IsoFLOP profiles) on TinyStories
   to find the compute-optimal model size and training length for a given
   budget.
5. **Containerisation.** Package training in a container, with data, checkpoints
   and MLflow tracking outside the image.
6. **Cloud training.** Run the larger IsoFLOP budgets and the final model on
   cloud GPUs.
7. **LoRA and QLoRA from scratch.** Fine-tune the trained model with both, and
   measure what 4-bit quantization costs, including in nats above the entropy
   floor on the artificial language.

## Use of AI

This project was built with the help of an AI coding assistant (Claude, used
through Claude Code). Because the project is partly a learning exercise, I wrote
the core parts myself and used the assistant mostly as a reviewer and tutor.

**My own work:**
- the project idea: benchmarking a transformer against the exact entropy of a
  probabilistic grammar
- the grammar
- the model components, adapted from a transformer I implemented earlier
- the Monte Carlo estimator
- the overall design of the data pipeline, tests and experiment
- the codebase architecture, design choices, contracts
- the testing infrastructure, e.g., fixtures, some tests but not all
- the choice of tools

**With help from the assistant:**
- **Code review.** I ported the model components from an implementation I wrote
  a few years ago and asked the assistant to review them for latent bugs. The
  review surfaced several silent ones—a few lines of code—such as unregistered
  submodules and a causal mask sized by the batch instead of the sequence.
  They are fixed, learned from, and the tests now guard against them.
- **Explanations** of concepts I was refreshing, such as the analytic entropy
  of a PCFG, `register_buffer` and specific pruning strategies in hyperparameter search.
- **Code.** I wrote the first versions of the training loop, the analytic
  estimator and the experiment script, including the Optuna search and MLflow
  tracking. The assistant refactored them and implemented parts I specified,
  such as pruning, tagging the best run, the final test evaluation and the
  linear-system solve. It also wrote some supporting code (the sampler that
  tracks log-probabilities, the vocabulary and collate functions) and several
  tests.
- **This README,** which it drafted and I edited.

The assistant also pointed out that the achievable loss per training window is
slightly below the per-token entropy, and that correction is now part of the
tests.

All results reported here come from running the code in this repository. I
reviewed the code and the explanations above and can account for them.
