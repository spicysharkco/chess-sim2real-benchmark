"""Label-free rules for combining default and flipped ChessQueries logits."""

import numpy as np

PIECE_SYMBOLS = ".PNBRQKpnbrqk"


def _probabilities(logits):
    """Return a stable softmax over the class axis."""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    weights = np.exp(shifted)
    return weights / np.sum(weights, axis=-1, keepdims=True)


def _structure_violations(predictions):
    """Score the fixed, deliberately incomplete chess-structure constraints."""
    scores = np.zeros(predictions.shape[:2], dtype=np.int64)

    white_kings = np.sum(predictions == 6, axis=-1)
    black_kings = np.sum(predictions == 12, axis=-1)
    scores += np.abs(white_kings - 1) + np.abs(black_kings - 1)

    white_pawns = np.sum(predictions == 1, axis=-1)
    black_pawns = np.sum(predictions == 7, axis=-1)
    scores += np.maximum(white_pawns - 8, 0)
    scores += np.maximum(black_pawns - 8, 0)

    back_ranks = predictions[..., np.r_[0:8, 56:64]]
    scores += np.sum((back_ranks == 1) | (back_ranks == 7), axis=-1)

    white_pieces = np.sum((predictions >= 1) & (predictions <= 6), axis=-1)
    black_pieces = np.sum((predictions >= 7) & (predictions <= 12), axis=-1)
    scores += np.maximum(white_pieces - 16, 0)
    scores += np.maximum(black_pieces - 16, 0)

    for sample, hypothesis in np.argwhere(
        (white_kings == 1) & (black_kings == 1)
    ):
        board = predictions[sample, hypothesis]
        white_square = int(np.flatnonzero(board == 6)[0])
        black_square = int(np.flatnonzero(board == 12)[0])
        white_rank, white_file = divmod(white_square, 8)
        black_rank, black_file = divmod(black_square, 8)
        if max(abs(white_rank - black_rank), abs(white_file - black_file)) == 1:
            scores[sample, hypothesis] += 1

    return scores


def _select(predictions, choices):
    sample_indices = np.arange(predictions.shape[0])
    return predictions[sample_indices, choices]


def predict_rules(logits):
    """Apply the preregistered label-free rules to two logit hypotheses.

    Args:
        logits: A finite NumPy array shaped ``[N, 2, 64, 13]``. Hypothesis 0
            is the released/default query-colour convention and hypothesis 1
            is the flipped convention.

    Returns:
        ``(predictions, diagnostics)``. Every prediction is an integer array
        shaped ``[N, 64]``. Choices use 0 for default and 1 for flipped.
    """
    if not isinstance(logits, np.ndarray):
        raise ValueError("logits must be a NumPy array")
    if logits.ndim != 4 or logits.shape[1:] != (2, 64, 13):
        raise ValueError("logits must have shape [N, 2, 64, 13]")
    if logits.shape[0] == 0:
        raise ValueError("logits must contain at least one sample")
    if not np.issubdtype(logits.dtype, np.number) or np.iscomplexobj(logits):
        raise ValueError("logits must contain real numeric values")
    if not np.isfinite(logits).all():
        raise ValueError("logits must contain only finite values")

    probabilities = _probabilities(logits)
    hypothesis_predictions = np.argmax(logits, axis=-1).astype(np.int64)
    max_probabilities = np.max(probabilities, axis=-1)
    confidence_scores = np.mean(np.log(max_probabilities), axis=-1)
    confidence_choice = (confidence_scores[:, 1] > confidence_scores[:, 0]).astype(
        np.int64
    )

    violations = _structure_violations(hypothesis_predictions)
    structure_choice = (
        (violations[:, 1] < violations[:, 0])
        | (
            (violations[:, 1] == violations[:, 0])
            & (confidence_scores[:, 1] > confidence_scores[:, 0])
        )
    ).astype(np.int64)

    predictions = {
        "default": hypothesis_predictions[:, 0].copy(),
        "flipped": hypothesis_predictions[:, 1].copy(),
        "confidence": _select(hypothesis_predictions, confidence_choice),
        "structure": _select(hypothesis_predictions, structure_choice),
        "mean_probability": np.argmax(
            np.mean(probabilities, axis=1), axis=-1
        ).astype(np.int64),
    }
    diagnostics = {
        "confidence_choice": confidence_choice,
        "structure_choice": structure_choice,
        "confidence_scores": confidence_scores,
        "violations": violations,
    }
    return predictions, diagnostics


def predict_guarded(logits):
    """Apply the derived valid-default guard around the original structure rule.

    A default prediction with zero violations under the declared structural
    checks is retained. Only an invalid default may be replaced by the original
    structure-then-confidence rule. This rule remains label-free and does not
    alter :func:`predict_rules` or its preregistered outputs.
    """
    predictions, original_diagnostics = predict_rules(logits)
    retained_valid_default = original_diagnostics["violations"][:, 0] == 0
    actual_choice = np.where(
        retained_valid_default,
        0,
        original_diagnostics["structure_choice"],
    ).astype(np.int64)
    guarded_predictions = np.where(
        retained_valid_default[:, None],
        predictions["default"],
        predictions["structure"],
    )
    diagnostics = {
        **original_diagnostics,
        "retained_valid_default": retained_valid_default,
        "actual_choice": actual_choice,
    }
    return guarded_predictions, diagnostics
