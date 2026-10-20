"""Route frozen boundary proposals to provisional training or listening review.

Character support and agreement are diagnostics, not calibrated confidence.
These rules never create human approvals."""

from __future__ import annotations

import math

def assess(case):
    reasons = []
    candidates = case.get("candidates", {})
    selection = case.get("selection", {})

    if selection.get("status") != "ok":
        reasons.append("selector_abstained")

    valid_engines = set()
    for engine in ("mms", "native"):
        candidate = candidates.get(engine, {})
        if candidate.get("status") != "ok" or candidate.get("stable") is not True:
            reasons.append(engine + "_unstable")

        valid = True
        for field in ("start", "end", "character_support", "endpoint_spread"):
            value = candidate.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                reasons.append(engine + "_invalid_" + field)
                valid = False

        if not valid:
            continue

        valid_engines.add(engine)
        if candidate["start"] < 0:
            reasons.append(engine + "_negative_start")
        if not 0 <= candidate["character_support"] <= 1:
            reasons.append(engine + "_invalid_support_range")
        if candidate["endpoint_spread"] < 0:
            reasons.append(engine + "_invalid_spread_range")
        if candidate.get("character_support", 0) < 0.50:
            reasons.append(engine + "_low_support")
        if candidate.get("endpoint_spread", float("inf")) > 0.06:
            reasons.append(engine + "_window_spread")
        if not 0.08 <= candidate.get("end", 0) - candidate.get("start", 0) <= 0.80:
            reasons.append(engine + "_duration")

    if len(valid_engines) == 2:
        disagreement = max(
            abs(candidates["mms"].get(key, 0) - candidates["native"].get(key, 0))
            for key in ("start", "end")
        )
        if disagreement > 0.08:
            reasons.append("model_disagreement")

    route = "auto_candidate" if not reasons else "review_required"
    return {
        "route": route,
        "reasons": reasons,
        "confidence_calibrated": False,
    }


def route_frozen_selection(case, duration):
    """Use the existing selector for provisional labels, never gold approvals.

    The stricter diagnostic gate only ranks confidence. It does not replace the
    frozen selector or turn a recovered v2 proposal into an automatic approval.
    """
    diagnostic = assess(case)
    selected = case.get("selection", {})
    bounds = selected.get("start"), selected.get("end")
    reasons = []

    def valid_number(value):
        return type(value) in (int, float) and math.isfinite(value)

    if selected.get("status") != "ok":
        reasons.append("frozen_selector_abstained")
    if (
        not valid_number(duration)
        or duration <= 0
        or not all(valid_number(value) for value in bounds)
    ):
        reasons.append("invalid_selected_bounds")
    elif not 0 <= bounds[0] < bounds[1] <= duration or bounds[1] - bounds[0] > 1:
        reasons.append("selected_word_does_not_fit_one_second")

    for engine in ("mms", "native"):
        candidate = case.get("candidates", {}).get(engine, {})
        if candidate.get("status") != "ok" or candidate.get("stable") is not True:
            reasons.append(engine + "_unusable")
        if (
            not valid_number(candidate.get("character_support"))
            or not 0.1 <= candidate["character_support"] <= 1
        ):
            reasons.append(engine + "_insufficient_support")
        if not all(
            valid_number(candidate.get(key))
            for key in ("start", "end", "endpoint_spread")
        ):
            reasons.append(engine + "_invalid_geometry")
        elif (
            not 0 <= candidate["start"] < candidate["end"] <= duration
            or not 0 <= candidate["endpoint_spread"] <= 0.12
        ):
            reasons.append(engine + "_invalid_geometry")

    if reasons:
        route = "review_required"
        confidence_band = "low"
    else:
        route = "auto_candidate"
        confidence_band = (
            "high" if diagnostic["route"] == "auto_candidate" else "standard"
        )

    return {
        "route": route,
        "label_source": "automatic_unreviewed",
        "human_reviewed": False,
        "confidence_band": confidence_band,
        "confidence_calibrated": False,
        "reasons": reasons,
        "diagnostic": diagnostic,
    }
