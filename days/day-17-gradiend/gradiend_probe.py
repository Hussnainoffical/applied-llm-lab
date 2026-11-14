"""Day 17: try the GRADIEND package (gradiend 0.2.5) on several features and models, CPU only.

One run per call, so each run gets a fresh process (a GRADIEND over all ~108M BERT weights needs a few GB):
    python gradiend_probe.py number_bert        # 3SG (he/she/it) vs 3PL (they), bert-base-cased
    python gradiend_probe.py number_bert_s1     # same, seed 1 (how stable is the top-k set?)
    python gradiend_probe.py gender_bert        # he vs she, bert-base-cased
    python gradiend_probe.py number_clinical    # 3SG vs 3PL on Bio_ClinicalBERT (a further-trained BERT)
    python gradiend_probe.py number_gpt2        # 3SG vs 3PL on distilgpt2, next-token objective
Writes results/<run>.json; compare.py reads those. Checkpoints go to %USERPROFILE%/gradiend-runs (not in git).
"""
import json, os, sys, time
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from transformers import AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from gradiend import TextFilterConfig, TextPredictionDataCreator, TextPredictionTrainer, TrainingArguments
from data import GENDER_EXTRA, NEUTRAL_EXCLUDE, NUMBER_TEXTS

NUMBER = dict(texts=NUMBER_TEXTS, classes=[(["he", "she", "it"], "3SG"), (["they"], "3PL")], rewrite="3PL",
              probe=["Yesterday after work [MASK] finished the report.", "When the bell rang [MASK] left the room.",
                     "After the meeting [MASK] went back to the office."], words=["he", "she", "it", "they"])
GENDER = dict(texts=NUMBER_TEXTS + GENDER_EXTRA, classes=[(["he"], "masc"), (["she"], "fem")], rewrite="fem",
              probe=["The engineer told us that [MASK] had fixed the bug.", "The nurse said that [MASK] would be late.",
                     "After dinner the professor said [MASK] was tired."], words=["he", "she"], min_left=4)
RUNS = {
    "number_bert": dict(model="bert-base-cased", feat=NUMBER, seed=0),
    "number_bert_s1": dict(model="bert-base-cased", feat=NUMBER, seed=1),
    "gender_bert": dict(model="bert-base-cased", feat=GENDER, seed=0),
    "number_clinical": dict(model="emilyalsentzer/Bio_ClinicalBERT", feat=NUMBER, seed=0),
    "number_gpt2": dict(model="distilgpt2", feat=NUMBER, seed=0, clm=True),
}


def probe(model, tok, sentences, words, clm):
    """Probability of each word at [MASK] (MLM) or as the next token after the text before [MASK] (CLM)."""
    model.eval(); out = {}
    for s in sentences:
        with torch.no_grad():
            if clm:
                enc = tok(s.split("[MASK]")[0].rstrip(), return_tensors="pt")
                p = model(**enc).logits[0, -1].softmax(-1)
            else:
                enc = tok(s.replace("[MASK]", tok.mask_token), return_tensors="pt")
                pos = (enc.input_ids[0] == tok.mask_token_id).nonzero()[0, 0]
                p = model(**enc).logits[0, pos].softmax(-1)
        ids = {w: tok.encode((" " if clm else "") + w, add_special_tokens=False)[0] for w in words}
        out[s] = {w: round(float(p[i]), 4) for w, i in ids.items()}
    return out


def main(name):
    cfg = RUNS[name]; feat = cfg["feat"]; clm = cfg.get("clm", False)
    torch.manual_seed(cfg["seed"])
    t0 = time.time()
    # min_left_context_words defaults to 10 in the creator (for MLM too); the short gender sentences need fewer
    creator = TextPredictionDataCreator(base_data=feat["texts"], min_left_context_words=feat.get("min_left", 10),
                                        feature_targets=[TextFilterConfig(targets=t, id=i) for t, i in feat["classes"]])
    training = creator.generate_training_data(max_size_per_class=60)
    neutral = creator.generate_neutral_data(additional_excluded_words=NEUTRAL_EXCLUDE, max_size=50)
    args = TrainingArguments(
        experiment_dir=str(Path.home() / "gradiend-runs" / name), train_batch_size=4, eval_steps=5, num_train_epochs=5,
        max_steps=50, learning_rate=1e-3, seed=cfg["seed"], fail_on_non_convergence=False,
        use_cache=os.environ.get("GRADIEND_USE_CACHE") == "1",  # reuse a finished checkpoint after an interrupted run
        max_seeds=1,  # shortcut: one seed per run on CPU; the default (3) silently retrains when the convergence check fails
        **({"prediction_objective": "clm_next_token"} if clm else {}))
    trainer = TextPredictionTrainer(model=cfg["model"], data=training, eval_neutral_data=neutral,
                                    eval_neutral_additional_excluded_words=NEUTRAL_EXCLUDE,
                                    max_counterfactuals_per_sentence=2, args=args)
    trainer.train()
    t_train = time.time() - t0
    enc = trainer.evaluate_encoder()
    dec = trainer.evaluate_decoder(target_class=feat["rewrite"], max_size=50)

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    base = (AutoModelForCausalLM if clm else AutoModelForMaskedLM).from_pretrained(cfg["model"])
    before = probe(base, tok, feat["probe"], feat["words"], clm); del base
    rewritten = trainer.rewrite_base_model(decoder_results=dec, target_class=feat["rewrite"])
    after = probe(rewritten, tok, feat["probe"], feat["words"], clm)

    meta = trainer.get_model().get_topk_feature_metadata(part="decoder-weight", topk=1000)
    stats = (trainer.get_training_stats() or {}).get("training_stats", {})
    n_train = {k: len(v) for k, v in training.items()} if isinstance(training, dict) else len(training)
    res = dict(run=name, model=cfg["model"], seed=cfg["seed"], n_train=n_train, n_neutral=len(neutral),
               classes=[i for _, i in feat["classes"]], rewrite_class=feat["rewrite"],
               encoder_correlation=enc.get("correlation"), training_correlation=stats.get("correlation"),
               decoder_summary=dec.get(feat["rewrite"]), probe_before=before, probe_after=after,
               topk_index=[m["base_global_index"] for m in meta], topk_param=[m["param_name"] for m in meta],
               train_seconds=round(t_train), total_seconds=round(time.time() - t0),
               versions=dict(torch=torch.__version__, gradiend=__import__("gradiend").__version__))
    Path("results").mkdir(exist_ok=True)
    Path(f"results/{name}.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("run", "encoder_correlation", "training_correlation", "train_seconds")}, default=str))


if __name__ == "__main__":
    main(sys.argv[1])
