#!/usr/bin/env python3
"""Create coauthor-ready figures from the frozen final result summaries."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "experiments" / "final_holdout" / "summaries"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "final_holdout" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT_DIR / ".mplconfig"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


COLORS = {
    "whisper": "#0072B2",
    "qwen3": "#D55E00",
    "speech_to_lyric": "#0072B2",
    "lyric_to_speech": "#D55E00",
    "neutral": "#4D4D4D",
    "good": "#009E73",
    "warning": "#E69F00",
    "bad": "#CC79A7",
}
MODEL_LABELS = {"whisper": "Whisper", "qwen3": "Qwen3-ASR"}


def load(name: str) -> dict[str, Any]:
    return json.loads((DATA_DIR / f"{name}.json").read_text())


def ci_error(row: dict[str, Any], key: str = "estimate") -> np.ndarray:
    center = float(row[key])
    return np.array(
        [[center - float(row["ci95_low"])], [float(row["ci95_high"]) - center]]
    )


def style_axis(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
    )


def save(fig: plt.Figure, stem: str) -> None:
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def validate_sources() -> None:
    """Fail before plotting if the frozen summary schema or scope changed."""
    e1, e3, e5, e6, e7, e8 = (
        load("e1"),
        load("e3"),
        load("e5"),
        load("e6_development"),
        load("e7"),
        load("e8"),
    )
    assert set(e1["models"]) == {"whisper", "qwen3"}
    e1_rows = [row for row in e1["primary_contrasts"] if row["split"] == "holdout"]
    assert len(e1_rows) == 12
    assert {(row["metric"], row["snr_db"]) for row in e1_rows} == {
        (metric, snr) for metric in ("lir", "tsr") for snr in (-10.0, -5.0, 0.0)
    }
    history_rows = [
        row for row in e3["history_effects"]["aggregate"] if row["split"] == "holdout"
    ]
    assert len(history_rows) == 16
    assert {row["condition"] for row in history_rows} == {"s_to_l", "l_to_s"}
    assert {row["switch_s"] for row in history_rows} == {0.5, 1.0, 2.0, 4.0}
    variants = {row["variant"] for row in e5["aggregates"]}
    assert variants == {"intact", "shuffle_500ms", "shuffle_100ms", "reverse", "instrumental"}
    assert e6["self_patch_mean_absolute_scs_deviation"] == 0.0
    assert [row["model"] for row in e7["evaluations"]] == [
        "M0_local",
        "M1_position",
        "M2_commitment",
    ]
    assert {row["condition"] for row in e8["aggregates"]} == {
        "baseline",
        "oracle_onset",
        "actual_onset",
        "actual_full",
        "actual_end",
    }
    for rows in (e1_rows, history_rows, e5["contrasts"], e8["contrasts"]):
        for row in rows:
            if "estimate" in row:
                assert row["ci95_low"] <= row["estimate"] <= row["ci95_high"]


def plot_core_story() -> None:
    e1, e3, e5 = load("e1"), load("e3"), load("e5")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.2))

    # A-B: preregistered E1 location contrasts, both primary metrics.
    snrs = [-10.0, -5.0, 0.0]
    offsets = {"whisper": -0.16, "qwen3": 0.16}
    metric_settings = {
        "lir": ("LIR(onset) − mean LIR(equal-budget locations)", "Negative favors onset"),
        "tsr": ("TSR(onset) − mean TSR(equal-budget locations)", "Positive favors onset"),
    }
    for panel, (ax, metric) in enumerate(zip(axes[0], ("lir", "tsr"))):
        rows = [
            row
            for row in e1["primary_contrasts"]
            if row["split"] == "holdout" and row["metric"] == metric
        ]
        for model in ("whisper", "qwen3"):
            selected = sorted(
                [row for row in rows if row["model"] == model],
                key=lambda row: row["snr_db"],
            )
            x = np.arange(len(snrs)) + offsets[model]
            y = np.array([row["estimate"] for row in selected])
            yerr = np.array(
                [
                    [row["estimate"] - row["ci95_low"] for row in selected],
                    [row["ci95_high"] - row["estimate"] for row in selected],
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                linewidth=1.5,
                markersize=6,
                color=COLORS[model],
                label=MODEL_LABELS[model],
            )
        ax.axhline(0, color=COLORS["neutral"], linewidth=1)
        ax.set_xticks(range(len(snrs)), [f"{int(snr)} dB" for snr in snrs])
        ax.set_ylabel(f"{metric_settings[metric][0]}\n({metric_settings[metric][1]})")
        ax.set_title(f"E1 holdout: {metric.upper()} location contrast")
        style_axis(ax)
        panel_label(ax, chr(ord("A") + panel))
    axes[0, 0].legend(frameon=False, loc="best")

    # C: E3 target-history contrast against the matched-static comparator.
    ax = axes[1, 0]
    rows = [
        row
        for row in e3["history_effects"]["aggregate"]
        if row["split"] == "holdout" and row["condition"] == "s_to_l"
    ]
    for model in ("whisper", "qwen3"):
        selected = sorted(
            [row for row in rows if row["model"] == model],
            key=lambda row: row["switch_s"],
        )
        x = np.array([row["switch_s"] for row in selected])
        y = np.array([row["estimate"] for row in selected])
        low = np.array([row["ci95_low"] for row in selected])
        high = np.array([row["ci95_high"] for row in selected])
        ax.plot(x, y, "o-", linewidth=1.8, color=COLORS[model], label=MODEL_LABELS[model])
        ax.fill_between(x, low, high, color=COLORS[model], alpha=0.14)
    ax.axhline(0, color=COLORS["neutral"], linewidth=1)
    ax.set_xticks([0.5, 1, 2, 4])
    ax.set_xlabel("Target-speech history (s)")
    ax.set_ylabel("SCS(S→L switch) − SCS(matched static)\n(negative = speech-history protection)")
    ax.set_title("E3 holdout: matched-local-acoustics contrast")
    ax.legend(frameon=False, loc="best")
    style_axis(ax)
    panel_label(ax, "C")

    # D: E5 lexical disruption contrasts.
    ax = axes[1, 1]
    order = ["shuffle_500ms", "shuffle_100ms", "reverse", "instrumental"]
    labels = ["500 ms shuffle", "100 ms shuffle", "Reverse", "Instrumental"]
    contrast_rows = {}
    for row in e5["contrasts"]:
        match = re.fullmatch(r"lir\((.+)\)-lir\(intact\)", row["contrast"])
        if match:
            contrast_rows[match.group(1)] = row
    selected = [contrast_rows[name] for name in order]
    y = np.arange(len(order))
    x = np.array([row["estimate"] for row in selected])
    xerr = np.array(
        [
            [row["estimate"] - row["ci95_low"] for row in selected],
            [row["ci95_high"] - row["estimate"] for row in selected],
        ]
    )
    ax.errorbar(
        x,
        y,
        xerr=xerr,
        fmt="o",
        capsize=3,
        color=COLORS["good"],
        linewidth=1.6,
        markersize=6,
    )
    ax.axvline(0, color=COLORS["neutral"], linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("LIR(disrupted) − LIR(intact)\n(negative = less capture)")
    ax.set_title("E5 holdout: paired lexical-disruption contrasts")
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "D")

    fig.suptitle("Selected final-holdout contrasts from E1, E3, and E5", fontsize=15)
    fig.text(
        0.5,
        -0.01,
        "Whiskers are paired track-bootstrap 95% CIs; available pair counts vary when outputs cannot be grounded.",
        ha="center",
        fontsize=9,
        color=COLORS["neutral"],
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94), w_pad=2.1, h_pad=2.5)
    save(fig, "figure_1_core_evidence")


def plot_history_asymmetry() -> None:
    e3 = load("e3")
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), sharey=True)
    condition_style = {
        "s_to_l": ("S→L: target history", "o-"),
        "l_to_s": ("L→S: lyric history", "s--"),
    }
    for ax, model, label in zip(axes, ("whisper", "qwen3"), ("Whisper", "Qwen3-ASR")):
        for condition in ("s_to_l", "l_to_s"):
            rows = sorted(
                [
                    row
                    for row in e3["history_effects"]["aggregate"]
                    if row["split"] == "holdout"
                    and row["model"] == model
                    and row["condition"] == condition
                ],
                key=lambda row: row["switch_s"],
            )
            x = np.array([row["switch_s"] for row in rows])
            y = np.array([row["estimate"] for row in rows])
            low = np.array([row["ci95_low"] for row in rows])
            high = np.array([row["ci95_high"] for row in rows])
            direction = "speech_to_lyric" if condition == "s_to_l" else "lyric_to_speech"
            name, fmt = condition_style[condition]
            ax.plot(x, y, fmt, color=COLORS[direction], linewidth=1.8, label=name)
            ax.fill_between(x, low, high, color=COLORS[direction], alpha=0.13)
        ax.axhline(0, color=COLORS["neutral"], linewidth=1)
        ax.set_xticks([0.5, 1, 2, 4])
        ax.set_xlabel("History duration (s)")
        ax.set_title(label)
        style_axis(ax)
    axes[0].set_ylabel("Switch SCS − matched-static SCS")
    axes[0].legend(frameon=False, loc="lower left")
    fig.suptitle("E3 holdout: direction-specific history contrasts", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.2)
    save(fig, "figure_2_history_asymmetry")


def plot_lexicality() -> None:
    e5 = load("e5")
    order = ["intact", "shuffle_500ms", "shuffle_100ms", "reverse", "instrumental"]
    labels = ["Intact", "500 ms shuffle", "100 ms shuffle", "Reverse", "Instrumental"]
    solo = {row["variant"]: row for row in e5["aggregates"] if row["mode"] == "solo"}
    mix = {row["variant"]: row for row in e5["aggregates"] if row["mode"] == "mix"}
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), sharex=True)
    x = np.arange(len(order))
    values = [solo[name]["mean_lyric_recall"] for name in order]
    axes[0].bar(x, values, color=COLORS["whisper"], alpha=0.86)
    axes[0].set_ylabel("Solo lyric recall")
    axes[0].set_title("Competitor intelligibility")
    values_mix = [mix[name]["mean_lir"] for name in order]
    axes[1].bar(x, values_mix, color=COLORS["good"], alpha=0.86)
    axes[1].set_ylabel("Mixed-audio mean LIR\n(grounded outputs only)")
    axes[1].set_title("Available-case source attribution")
    values_missing = [mix[name]["mean_no_grounded_output"] for name in order]
    axes[2].bar(x, values_missing, color=COLORS["warning"], alpha=0.86)
    axes[2].set_ylabel("No-grounded-output rate")
    axes[2].set_title("Missing-output sensitivity")
    for ax, values_here in zip(axes, (values, values_mix, values_missing)):
        ax.set_xticks(x, labels, rotation=24, ha="right")
        ax.set_ylim(0, max(values_here) * 1.22)
        for xpos, value in zip(x, values_here):
            ax.text(xpos, value + max(values_here) * 0.035, f"{value:.2f}", ha="center", fontsize=9)
        style_axis(ax)
    fig.suptitle("E5 holdout descriptive means; paired LIR contrasts are shown in Figure 1D", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.4)
    save(fig, "figure_3_lexicality")


def plot_e4_caveat() -> None:
    e4a, e4b = load("e4a"), load("e4b")
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.7))

    ax = axes[0]
    strata = ["boundary", "stable_speech", "stable_lyric"]
    prefixes = ["none", "speech", "lyric", "unrelated"]
    label_map = {"boundary": "Boundary", "stable_speech": "Stable speech", "stable_lyric": "Stable lyric"}
    colors = [COLORS["neutral"], COLORS["whisper"], COLORS["qwen3"], COLORS["warning"]]
    width = 0.19
    x = np.arange(len(strata))
    for index, (prefix, color) in enumerate(zip(prefixes, colors)):
        rows = {row["selection_stratum"]: row for row in e4a["aggregates"] if row["prefix_condition"] == prefix}
        values = [rows[stratum]["mean_no_grounded_output"] for stratum in strata]
        ax.bar(x + (index - 1.5) * width, values, width, label=prefix.title(), color=color, alpha=0.86)
    ax.set_xticks(x, [label_map[name] for name in strata])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("No-grounded-output rate")
    ax.set_title("Prefix interventions often stop decoding")
    ax.legend(frameon=False, ncol=2, fontsize=9)
    style_axis(ax)
    panel_label(ax, "A")

    ax = axes[1]
    rows = [
        row
        for row in e4b["contrasts_zero_filled"]
        if row["contrast"] == "SCS(carry_natural)-SCS(reset)"
    ]
    offsets = {"s_to_l": -0.08, "l_to_s": 0.08}
    labels = {"s_to_l": "S→L", "l_to_s": "L→S"}
    directions = {"s_to_l": "speech_to_lyric", "l_to_s": "lyric_to_speech"}
    for direction in ("s_to_l", "l_to_s"):
        selected = sorted([row for row in rows if row["stratum_or_direction"] == direction], key=lambda row: row["switch_s"])
        xvals = np.array([row["switch_s"] for row in selected]) + offsets[direction]
        yvals = np.array([row["estimate"] for row in selected])
        yerr = np.array(
            [
                [row["estimate"] - row["ci95_low"] for row in selected],
                [row["ci95_high"] - row["estimate"] for row in selected],
            ]
        )
        ax.errorbar(
            xvals,
            yvals,
            yerr=yerr,
            fmt="o-",
            capsize=3,
            linewidth=1.5,
            color=COLORS[directions[direction]],
            label=labels[direction],
        )
    ax.axhline(0, color=COLORS["neutral"], linewidth=1)
    ax.set_xticks([1, 2, 4])
    ax.set_xlabel("History duration (s)")
    ax.set_ylabel("Carry-natural − reset SCS\n(zero-filled)")
    ax.set_title("Observed carry-natural minus reset contrasts")
    ax.legend(frameon=False)
    style_axis(ax)
    panel_label(ax, "B")

    fig.suptitle("E4 holdout: observed history contrasts and termination rates", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.4)
    save(fig, "figure_4_e4_caveat")


def plot_negative_results() -> None:
    e7, e8 = load("e7"), load("e8")
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5))

    ax = axes[0]
    evaluations = e7["evaluations"]
    labels = ["M0 local", "M1 position", "M2 commitment"]
    values = [row["heldout_token_nll"] for row in evaluations]
    colors = [COLORS["neutral"], COLORS["whisper"], COLORS["bad"]]
    deltas = np.array(values) - values[0]
    ypos = np.arange(len(labels))
    ax.scatter(deltas, ypos, c=colors, s=48, zorder=3)
    ax.axvline(0, color=COLORS["neutral"], linewidth=1)
    ax.set_yticks(ypos, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Held-out token NLL − M0 NLL\n(negative = better)")
    ax.set_title("E7: relative held-out fit")
    ax.set_xlim(min(-0.001, float(deltas.min()) - 0.001), float(deltas.max()) + 0.001)
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value:.3f}")
    for xpos, ypos_value, value in zip(deltas, ypos, values):
        ax.text(xpos + 0.001, ypos_value, f"NLL={value:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "A")

    ax = axes[1]
    order = ["oracle_onset", "actual_onset", "actual_full", "actual_end"]
    labels = ["Oracle onset", "Actual onset", "Actual full", "Actual end"]
    metric_style = {"lir": ("LIR improvement", "o", COLORS["good"]), "tsr": ("TSR improvement", "s", COLORS["whisper"])}
    parsed: dict[tuple[str, str], dict[str, Any]] = {}
    for row in e8["contrasts"]:
        match = re.fullmatch(r"(lir|tsr)\((.+)\)-(?:lir|tsr)\(baseline\)", row["contrast"])
        if match:
            parsed[(match.group(1), match.group(2))] = row
    ybase = np.arange(len(order))
    for metric, (name, marker, color) in metric_style.items():
        offset = -0.09 if metric == "lir" else 0.09
        centers, lows, highs = [], [], []
        for condition in order:
            row = parsed[(metric, condition)]
            if metric == "lir":
                centers.append(-row["estimate"])
                lows.append(-row["ci95_high"])
                highs.append(-row["ci95_low"])
            else:
                centers.append(row["estimate"])
                lows.append(row["ci95_low"])
                highs.append(row["ci95_high"])
        centers_arr = np.array(centers)
        xerr = np.array([centers_arr - np.array(lows), np.array(highs) - centers_arr])
        ax.errorbar(
            centers_arr,
            ybase + offset,
            xerr=xerr,
            fmt=marker,
            capsize=3,
            color=color,
            linewidth=1.5,
            markersize=6,
            label=name,
        )
    ax.axvline(0, color=COLORS["neutral"], linewidth=1)
    ax.set_yticks(ybase, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Improvement vs baseline (positive = better)")
    ax.set_title("E8: oracle works; onset-only frontend does not")
    ax.legend(frameon=False)
    ax.grid(axis="x", color="#D0D0D0", linewidth=0.7, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    panel_label(ax, "B")

    fig.suptitle("Final holdout: E7 and E8 tests that did not support the proposed model or frontend", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=2.6)
    save(fig, "figure_5_negative_results")


def plot_patching() -> None:
    e6 = load("e6_development")
    cells = e6["normalized_cells"]
    layers = sorted({row["layer"] for row in cells})
    positions = ["onset", "middle", "offset"]
    directions = ["speech_to_lyric", "lyric_to_speech"]
    titles = ["Speech donor → lyric receiver", "Lyric donor → speech receiver"]
    matrices = []
    for direction in directions:
        lookup = {(row["position"], row["layer"]): row for row in cells if row["direction"] == direction}
        matrices.append(
            np.array(
                [
                    [lookup[(position, layer)]["stable_denominator_estimate"] for layer in layers]
                    for position in positions
                ]
            )
        )
    vmax = max(float(np.max(matrix)) for matrix in matrices)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 3.8), constrained_layout=True)
    image = None
    for ax, matrix, title in zip(axes, matrices, titles):
        image = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(layers)), layers)
        ax.set_yticks(range(len(positions)), [name.title() for name in positions])
        ax.set_xlabel("Encoder layer")
        ax.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                text_color = "white" if value > vmax * 0.55 else "black"
                ax.text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=8, color=text_color)
    axes[0].set_ylabel("Patched temporal position")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.88)
    colorbar.set_label("Normalized SCS transfer")
    fig.suptitle("E6 development-only activation patching (exploratory)", fontsize=14)
    save(fig, "figure_6_activation_patching")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    validate_sources()
    plot_core_story()
    plot_history_asymmetry()
    plot_lexicality()
    plot_e4_caveat()
    plot_negative_results()
    plot_patching()
    print(f"Wrote figures to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
