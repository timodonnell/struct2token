"""Published baseline results for structure tokenization methods.

Numbers are sourced from the respective papers:
- APT: Adaptive Protein Tokenization (Table 1)
- Bio2Token: All-atom tokenization (Table 1)
- Kanzi: Protein structure tokenization (Table 1)

All RMSD/RMSE values are in Angstroms. TM-scores are unitless [0, 1].
"""

from __future__ import annotations

from typing import Any, Dict, Optional

# Published results keyed by (method, dataset, metric)
# None means the metric was not reported for that dataset
BASELINES: Dict[str, Dict[str, Dict[str, Optional[float]]]] = {
    "apt": {
        # APT reports Cα-only RMSD and TM-score
        # From Table 1, using 16 tokens (their default)
        "cath": {"ca_rmsd": 0.90, "tm_score": 0.941, "all_atom_rmsd": None},
        "cameo": {"ca_rmsd": 0.90, "tm_score": 0.941, "all_atom_rmsd": None},
        "afdb": {"ca_rmsd": 1.17, "tm_score": 0.929, "all_atom_rmsd": None},
        "casp14": {"ca_rmsd": None, "tm_score": None, "all_atom_rmsd": None},
        "casp15": {"ca_rmsd": None, "tm_score": None, "all_atom_rmsd": None},
    },
    "bio2token": {
        # Bio2Token reports all-atom RMSE (equivalent to RMSD)
        # From Table 1
        "cath4.2": {
            "all_atom_rmsd": 0.56,
            "all_atom_rmsd_std": 0.06,
            "ca_rmsd": None,
            "tm_score": None,
        },
        "casp14": {
            "all_atom_rmsd": 0.58,
            "all_atom_rmsd_std": 0.10,
            "ca_rmsd": None,
            "tm_score": None,
        },
        "casp15": {
            "all_atom_rmsd": 0.59,
            "all_atom_rmsd_std": 0.11,
            "ca_rmsd": None,
            "tm_score": None,
        },
    },
    "kanzi": {
        # Kanzi reports Cα RMSD and TM-score
        # From Table 1
        "cameo": {"ca_rmsd": 0.936, "tm_score": 0.948, "all_atom_rmsd": None},
        "casp14": {"ca_rmsd": 0.861, "tm_score": 0.958, "all_atom_rmsd": None},
        "casp15": {"ca_rmsd": 1.345, "tm_score": 0.951, "all_atom_rmsd": None},
        "cath": {"ca_rmsd": 1.098, "tm_score": 0.940, "all_atom_rmsd": None},
    },
}

# Method descriptions for table headers
METHOD_INFO = {
    "apt": {"name": "APT", "representation": "Cα-only", "n_tokens": 16},
    "bio2token": {"name": "Bio2Token", "representation": "All-atom", "n_tokens": "variable"},
    "kanzi": {"name": "Kanzi", "representation": "Cα/backbone", "n_tokens": 64},
    "struct2token": {"name": "Struct2Token", "representation": "All-atom", "n_tokens": "adaptive"},
}


def format_comparison_table(
    struct2token_results: Dict[str, Dict[str, float]],
    datasets: list[str] | None = None,
    n_tokens_list: list[int] | None = None,
) -> str:
    """Format a markdown comparison table of struct2token vs published baselines.

    Args:
        struct2token_results: Dict keyed by dataset, then by metric name.
            Metric names: ca_rmsd, all_atom_rmsd, tm_score (all in Angstroms / unitless).
            Can also be keyed by f"{dataset}_n{n_tokens}" for token-sweep results.
        datasets: Which datasets to include. Defaults to ["casp14", "casp15"].
        n_tokens_list: If provided, show struct2token results at each token count.

    Returns:
        Markdown-formatted comparison table string.
    """
    if datasets is None:
        datasets = ["casp14", "casp15"]
    if n_tokens_list is None:
        n_tokens_list = [128]

    lines = []

    # --- Table 1: Cα RMSD comparison ---
    lines.append("## Cα RMSD (Å) Comparison")
    lines.append("")

    header = "| Method | Repr. |"
    sep = "|--------|-------|"
    for ds in datasets:
        header += f" {ds.upper()} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # Published baselines
    for method_key in ["apt", "bio2token", "kanzi"]:
        info = METHOD_INFO[method_key]
        row = f"| {info['name']} | {info['representation']} |"
        for ds in datasets:
            val = _get_baseline(method_key, ds, "ca_rmsd")
            row += f" {val} |"
        lines.append(row)

    # struct2token at each token count
    for n_tok in n_tokens_list:
        suffix = f"_n{n_tok}" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        tok_label = f" ({n_tok} tok)" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        row = f"| **Struct2Token**{tok_label} | All-atom |"
        for ds in datasets:
            key = f"{ds}{suffix}" if suffix else ds
            val = struct2token_results.get(key, {}).get("ca_rmsd")
            row += f" {_fmt(val)} |"
        lines.append(row)

    lines.append("")

    # --- Table 2: All-atom RMSD comparison ---
    lines.append("## All-Atom RMSD (Å) Comparison")
    lines.append("")

    header = "| Method | Repr. |"
    sep = "|--------|-------|"
    for ds in datasets:
        header += f" {ds.upper()} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for method_key in ["apt", "bio2token", "kanzi"]:
        info = METHOD_INFO[method_key]
        row = f"| {info['name']} | {info['representation']} |"
        for ds in datasets:
            val = _get_baseline(method_key, ds, "all_atom_rmsd")
            row += f" {val} |"
        lines.append(row)

    for n_tok in n_tokens_list:
        suffix = f"_n{n_tok}" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        tok_label = f" ({n_tok} tok)" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        row = f"| **Struct2Token**{tok_label} | All-atom |"
        for ds in datasets:
            key = f"{ds}{suffix}" if suffix else ds
            val = struct2token_results.get(key, {}).get("all_atom_rmsd")
            row += f" {_fmt(val)} |"
        lines.append(row)

    lines.append("")

    # --- Table 3: TM-score comparison ---
    lines.append("## TM-Score Comparison")
    lines.append("")

    header = "| Method | Repr. |"
    sep = "|--------|-------|"
    for ds in datasets:
        header += f" {ds.upper()} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for method_key in ["apt", "bio2token", "kanzi"]:
        info = METHOD_INFO[method_key]
        row = f"| {info['name']} | {info['representation']} |"
        for ds in datasets:
            val = _get_baseline(method_key, ds, "tm_score")
            row += f" {val} |"
        lines.append(row)

    for n_tok in n_tokens_list:
        suffix = f"_n{n_tok}" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        tok_label = f" ({n_tok} tok)" if len(n_tokens_list) > 1 or n_tok != 128 else ""
        row = f"| **Struct2Token**{tok_label} | All-atom |"
        for ds in datasets:
            key = f"{ds}{suffix}" if suffix else ds
            val = struct2token_results.get(key, {}).get("tm_score")
            row += f" {_fmt(val, decimals=3)} |"
        lines.append(row)

    lines.append("")

    # --- Table 4: Token-count sweep (if multiple) ---
    if len(n_tokens_list) > 1:
        lines.append("## Token Count Sweep (Struct2Token)")
        lines.append("")
        header = "| n_tokens |"
        sep = "|----------|"
        for ds in datasets:
            header += f" Cα RMSD ({ds.upper()}) | AA RMSD ({ds.upper()}) | TM ({ds.upper()}) |"
            sep += "--------|--------|--------|"
        lines.append(header)
        lines.append(sep)

        for n_tok in n_tokens_list:
            row = f"| {n_tok} |"
            for ds in datasets:
                key = f"{ds}_n{n_tok}"
                r = struct2token_results.get(key, {})
                row += f" {_fmt(r.get('ca_rmsd'))} |"
                row += f" {_fmt(r.get('all_atom_rmsd'))} |"
                row += f" {_fmt(r.get('tm_score'), decimals=3)} |"
            lines.append(row)
        lines.append("")

    return "\n".join(lines)


def _get_baseline(method: str, dataset: str, metric: str) -> str:
    """Look up a baseline value, returning formatted string or '-'."""
    method_data = BASELINES.get(method, {})
    ds_data = method_data.get(dataset, {})
    val = ds_data.get(metric)
    if val is None:
        return "-"
    std = ds_data.get(f"{metric}_std")
    if std is not None:
        return f"{val:.2f}±{std:.2f}"
    return f"{val:.3f}" if metric == "tm_score" else f"{val:.2f}"


def _fmt(val: float | None, decimals: int = 2) -> str:
    """Format a numeric value or return '-' for None."""
    if val is None:
        return "-"
    return f"{val:.{decimals}f}"
