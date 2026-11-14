# Day 17: trying the GRADIEND package (gradient-based feature learning)

GRADIEND ([paper](https://arxiv.org/abs/2502.01406), [package paper](https://arxiv.org/abs/2602.23993),
`pip install gradiend`, v0.2.5) learns where a language model encodes a feature: an encoder squeezes the
gradient of a masked-token prediction into one number in [-1, 1], a decoder maps that number back to a
weight update, and the decoder can rewrite the base model to push the feature up or down.
I ran it on CPU only (torch 2.15 CPU, Windows) with small data, on two features, three base models and
two seeds, to see what it learns and where the defaults bite.

## Setup

- Number: 3SG (he / she / it) vs 3PL (they), the 75 sentences from the package's `start_workflow` example
  (Apache-2.0, in `data.py`), giving 14 + 19 masked training rows.
- Gender: he vs she, those 75 plus 40 sentences I wrote, giving 31 + 29 rows (`min_left_context_words=4`).
- 50 steps, batch 4, lr 1e-3, GRADIEND over all ~108M BERT weights (heads excluded), one seed per run.
- After training: encoder evaluation, decoder grid (strengthen one class), `rewrite_base_model`, then my
  own probe sentences scored on the base and the rewritten model, and the top-1000 decoder weights.

## Results (measured 14 Nov 2026)

| Run | Base model | Rows | Best train corr. | Converged | Rewrite (lr) | LMS before / after |
|---|---|---|---|---|---|---|
| number_bert | bert-base-cased | 14 / 19 | 0.990 (step 10) | yes | strengthen 3PL (0.2) | 0.515 / 0.515 |
| gender_bert | bert-base-cased | 31 / 29 | 0.998 (step 50) | yes | strengthen fem (0.1) | 0.455 / 0.455 |
| number_gpt2 | distilgpt2 (next-token) | 14 / 19 | 1.000 (step 30+) | yes | strengthen 3PL (0.2) | 0.192 / 0.191 |

Rewrite effect on sentences the model never saw (probability of the word at `[MASK]`):

| Sentence | Before | After |
|---|---|---|
| After dinner the professor said [MASK] was tired. (gender, fem) | he 0.890, she 0.077 | he 0.002, she 0.885 |
| The engineer told us that [MASK] had fixed the bug. (gender, fem) | he 0.309, she 0.030 | he 0.005, she 0.602 |
| After the meeting [MASK] went back to the office. (number, 3PL) | they 0.153, he 0.220 | they 0.168, he 0.250 |
| When the bell rang [MASK] left the room. (number, 3PL) | they 0.054, he 0.242 | they 0.078, he 0.281 |
| Yesterday after work → next token (distilgpt2, 3PL) | they 0.003, he 0.009 | they 0.046, he 0.011 |
| After the meeting → next token (distilgpt2, 3PL) | they 0.003, he 0.004 | they 0.018, he 0.006 |

Where the top-1000 decoder weights sit: gender 411 in the embeddings and 492 in layers 8 to 11;
number 359 in the embeddings and 479 in layers 9 to 11 (mostly attention value matrices). The two
features share 14% of their top-1000 weights. In distilgpt2 (6 layers) number sits in layer 4 (301),
layer 5 (180) and the embeddings (270).

## What I take from it

- The gender direction is clean and strong: one decoder step moves "the professor said [MASK]" from 89% he
  to 89% she, and the language-model score on neutral text does not move. For debiasing you would pick a
  smaller step than the "strengthen" optimum.
- The number direction trained just as well by correlation, but the rewrite is weak and not selective:
  "they" rose by 1 to 2 points while "he" rose by 3 to 4. With 14 + 19 rows it learned "subject pronoun"
  more than "plural". The same data on distilgpt2 gave a more selective edit: the next-token "they" rose
  6 to 15 fold while "he" and "she" barely moved (from small base probabilities).
- Both features live mostly in token embeddings and the last four layers, but in different weights (14%
  overlap), so they could be edited separately.

## Gotchas I hit (worth knowing before you use it)

1. `TextPredictionDataCreator` defaults to `min_left_context_words=10` for every model, MLM included. With
   short sentences that silently drops most matches: only 2 "he" rows survived from 115 sentences, and
   `train()` then failed with `per_class_dict_to_unified produced no rows`. Set it explicitly.
2. The best checkpoint is picked by encoder correlation on a small validation split (3 to 6 rows here).
   That can pick an early step whose class means are still small. A second bert-base seed peaked at step 5
   (corr 0.997, mean 3SG -0.28), failed the 0.5 mean check, and the default `max_seeds=3` started a new
   seed, while its own step-45 checkpoint (mean -0.68 / 1.0) would have passed. On Bio_ClinicalBERT the
   untrained step 0 scored 0.993 and stayed "best" through step 15 (0.91, 0.80, 0.68 after it); on distilgpt2
   step 0 scored 0.9999 and was only beaten at step 20.
3. Runs were cut short by a lost connection, so the clinical-BERT and second-seed comparisons are
   incomplete; only the observations above are reported from their logs.

Run: `python gradiend_probe.py number_bert` (or gender_bert, number_bert_s1, number_clinical, number_gpt2),
then `python compare.py` · Test: `python -m pytest -q` (pure-Python helpers only) · needs torch,
transformers and gradiend in a venv; checkpoints go to `%USERPROFILE%\gradiend-runs`, not to git.
