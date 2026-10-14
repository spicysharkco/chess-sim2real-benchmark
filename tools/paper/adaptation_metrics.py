"""Pure-NumPy scoring and preregistered selection for the transfer study."""

from collections.abc import Mapping

import numpy as np

CONDITIONS = ("control", "parity")
RULE_ORDER = ("confidence", "structure", "mean_probability")


def _validated_inputs(pred, labels, records):
    if not isinstance(pred, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("pred and labels must be NumPy arrays")
    if pred.ndim != 2 or pred.shape[1:] != (64,):
        raise ValueError("pred must have shape [N, 64]")
    if labels.shape != pred.shape:
        raise ValueError("labels must have the same [N, 64] shape as pred")
    if pred.shape[0] == 0:
        raise ValueError("pred and labels must contain at least one board")
    for name, values in (("pred", pred), ("labels", labels)):
        if not np.issubdtype(values.dtype, np.integer) or np.issubdtype(
            values.dtype, np.bool_
        ):
            raise ValueError(f"{name} must contain integer class indices")
        if np.any((values < 0) | (values > 12)):
            raise ValueError(f"{name} class indices must be in [0, 12]")

    try:
        rows = list(records)
    except TypeError as error:
        raise ValueError("records must be an iterable with one row per board") from error
    if len(rows) != pred.shape[0]:
        raise ValueError("records must contain one row per board")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each record must be a mapping")
        if not {"condition", "family", "group_id"}.issubset(row):
            raise ValueError("records require condition, family, and group_id")
        if row["condition"] not in CONDITIONS:
            raise ValueError("record condition must be 'control' or 'parity'")
        for field in ("family", "group_id"):
            try:
                hash(row[field])
            except TypeError as error:
                raise ValueError(f"record {field} must be hashable") from error
    return rows


def _metrics(pred, labels, indices):
    selected_pred = pred[indices]
    selected_labels = labels[indices]
    correct = selected_pred == selected_labels
    exact = np.all(correct, axis=1)
    n = int(len(indices))
    exact_boards = int(np.sum(exact))
    return {
        "n": n,
        "exact_boards": exact_boards,
        "exact_pct": 100.0 * exact_boards / n,
        "per_square_pct": 100.0 * float(np.sum(correct)) / (n * 64),
    }


def score_predictions(pred, labels, records):
    """Score board predictions overall and by registered condition and family."""
    rows = _validated_inputs(pred, labels, records)
    all_indices = np.arange(pred.shape[0])
    condition_indices = {
        condition: np.array(
            [i for i, row in enumerate(rows) if row["condition"] == condition]
        )
        for condition in CONDITIONS
        if any(row["condition"] == condition for row in rows)
    }

    families = dict.fromkeys(row["family"] for row in rows)
    family_indices = {
        family: np.array(
            [i for i, row in enumerate(rows) if row["family"] == family]
        )
        for family in families
    }
    return {
        "overall": _metrics(pred, labels, all_indices),
        "by_condition": {
            condition: _metrics(pred, labels, indices)
            for condition, indices in condition_indices.items()
        },
        "by_family": {
            family: _metrics(pred, labels, indices)
            for family, indices in family_indices.items()
        },
    }


def _metric(score, section, key=None):
    try:
        value = score[section] if key is None else score[section][key]
        return value
    except (KeyError, TypeError) as error:
        raise ValueError("score has an invalid metrics structure") from error


def _condition_square_safe(candidate, baseline):
    for condition in CONDITIONS:
        candidate_metric = _metric(candidate, "by_condition", condition)
        baseline_metric = _metric(baseline, "by_condition", condition)
        if candidate_metric["per_square_pct"] < baseline_metric["per_square_pct"] - 1:
            return False
    return True


def choose_rule(scores):
    """Choose the best validation-eligible label-free rule, or ``None``."""
    required = {"default", *RULE_ORDER}
    if not isinstance(scores, Mapping) or not required.issubset(scores):
        raise ValueError("scores must contain default and every registered rule")
    baseline = scores["default"]
    baseline_overall = _metric(baseline, "overall")
    eligible = []
    for rule in RULE_ORDER:
        candidate = scores[rule]
        overall = _metric(candidate, "overall")
        if overall["per_square_pct"] < baseline_overall["per_square_pct"] + 5:
            continue
        if overall["exact_pct"] < baseline_overall["exact_pct"]:
            continue
        if not _condition_square_safe(candidate, baseline):
            continue
        eligible.append(rule)
    if not eligible:
        return None
    return max(eligible, key=lambda rule: _metric(scores[rule], "overall")["per_square_pct"])


def choose_adapter(scores, parameter_counts):
    """Choose the best validation-eligible adapter, or ``None``."""
    if not isinstance(scores, Mapping) or "default" not in scores:
        raise ValueError("scores must contain default")
    if not isinstance(parameter_counts, Mapping):
        raise ValueError("parameter_counts must be a mapping")
    baseline = scores["default"]
    baseline_overall = _metric(baseline, "overall")
    best_name = None
    best_rank = None
    for name, candidate in scores.items():
        if name == "default":
            continue
        if name not in parameter_counts or parameter_counts[name] < 0:
            raise ValueError(f"missing valid parameter count for {name}")
        overall = _metric(candidate, "overall")
        if overall["exact_pct"] < baseline_overall["exact_pct"] + 5:
            continue
        if not _condition_square_safe(candidate, baseline):
            continue
        rank = (
            overall["exact_pct"],
            overall["per_square_pct"],
            -parameter_counts[name],
        )
        if best_rank is None or rank > best_rank:
            best_name = name
            best_rank = rank
    return best_name


def paired_summary(pred, baseline, labels, records, seed=20260910, draws=2000):
    """Return group-paired, family-stratified bootstrap deltas and intervals."""
    rows = _validated_inputs(pred, labels, records)
    _validated_inputs(baseline, labels, rows)
    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise ValueError("draws must be a positive integer")

    candidate_exact = np.all(pred == labels, axis=1)
    baseline_exact = np.all(baseline == labels, axis=1)
    candidate_square = np.mean(pred == labels, axis=1)
    baseline_square = np.mean(baseline == labels, axis=1)
    exact_delta = candidate_exact.astype(float) - baseline_exact.astype(float)
    square_delta = candidate_square - baseline_square

    group_rows = {}
    for index, row in enumerate(rows):
        group_key = (row["family"], row["group_id"])
        group_rows.setdefault(group_key, []).append(index)

    family_groups = {}
    group_deltas = {}
    for group_key, indices in group_rows.items():
        family_groups.setdefault(group_key[0], []).append(group_key)
        group_deltas[group_key] = (
            float(np.mean(exact_delta[indices])),
            float(np.mean(square_delta[indices])),
        )

    point_exact = 100.0 * np.mean([delta[0] for delta in group_deltas.values()])
    point_square = 100.0 * np.mean([delta[1] for delta in group_deltas.values()])
    rng = np.random.default_rng(seed)
    bootstrap_exact = np.empty(draws)
    bootstrap_square = np.empty(draws)
    for draw in range(draws):
        sampled = []
        for groups in family_groups.values():
            sampled.extend(
                groups[index]
                for index in rng.integers(0, len(groups), size=len(groups))
            )
        bootstrap_exact[draw] = 100.0 * np.mean(
            [group_deltas[group_id][0] for group_id in sampled]
        )
        bootstrap_square[draw] = 100.0 * np.mean(
            [group_deltas[group_id][1] for group_id in sampled]
        )

    return {
        "exact_delta_pp": float(point_exact),
        "square_delta_pp": float(point_square),
        "exact_ci95_pp": np.percentile(bootstrap_exact, [2.5, 97.5]).tolist(),
        "square_ci95_pp": np.percentile(bootstrap_square, [2.5, 97.5]).tolist(),
        "gained": int(np.sum(candidate_exact & ~baseline_exact)),
        "lost": int(np.sum(~candidate_exact & baseline_exact)),
    }


def confirm_pass(kind, score, baseline_score, paired):
    """Evaluate the preregistered confirmation gates with explicit reasons."""
    if kind not in {"adapter", "rule"}:
        raise ValueError("kind must be 'adapter' or 'rule'")
    reasons = []
    try:
        exact_delta = paired["exact_delta_pp"]
        square_delta = paired["square_delta_pp"]
        exact_ci_lower = paired["exact_ci95_pp"][0]
        square_ci_lower = paired["square_ci95_pp"][0]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("paired has an invalid summary structure") from error

    if kind == "adapter":
        if exact_ci_lower <= 0:
            reasons.append("exact_ci_lower_not_positive")
        if square_delta <= 0:
            reasons.append("square_delta_not_positive")
    else:
        if square_delta < 5:
            reasons.append("square_delta_below_5pp")
        if square_ci_lower <= 0:
            reasons.append("square_ci_lower_not_positive")
        if exact_delta < 0:
            reasons.append("exact_loss")

    parity = _metric(score, "by_condition", "parity")
    baseline_parity = _metric(baseline_score, "by_condition", "parity")
    if parity["per_square_pct"] < baseline_parity["per_square_pct"] - 1:
        reasons.append("parity_square_loss_over_1pp")

    score_families = _metric(score, "by_family")
    baseline_families = _metric(baseline_score, "by_family")
    for family, baseline_family in baseline_families.items():
        if family not in score_families:
            reasons.append(f"missing_family:{family}")
        elif (
            score_families[family]["per_square_pct"]
            < baseline_family["per_square_pct"] - 2
        ):
            reasons.append(f"family_square_loss_over_2pp:{family}")
    return {"passed": not reasons, "reasons": reasons}
