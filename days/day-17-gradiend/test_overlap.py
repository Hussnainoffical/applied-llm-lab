from overlap import topk_overlap, where


def test_overlap():
    assert topk_overlap([1, 2, 3, 4], [3, 4, 5, 6]) == 0.5
    assert topk_overlap([1, 2], [1, 2, 3, 4]) == 1.0
    assert topk_overlap([], [1]) == 0.0


def test_where():
    names = ["bert.encoder.layer.11.output.dense.weight", "bert.encoder.layer.2.attention.self.query.weight",
             "bert.embeddings.word_embeddings.weight", "transformer.h.5.mlp.c_fc.weight", "cls.predictions.bias"]
    assert where(names) == {"embeddings": 1, "head/other": 1, "layer 02": 1, "layer 05": 1, "layer 11": 1}
