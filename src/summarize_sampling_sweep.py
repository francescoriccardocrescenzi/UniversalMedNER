import argparse
import json
from pathlib import Path

# (metrics key, column label, higher_is_better)
# F1_soft/F1_strict are corpus-level: TP/GT/PRED are summed across every sample
# before a single F1 is computed, i.e. micro-averaged. The reward_*_mean fields
# are the mean of a per-sample score computed independently for each example,
# i.e. macro-averaged. The two aggregations answer different questions and are
# not expected to match even when they look at "the same" underlying quantity.
METRIC_COLUMNS = [
    ("F1_soft", "F1 soft (micro)", True),
    ("F1_strict", "F1 strict (micro)", True),
    ("reward_soft_f1_mean", "Soft-F1 reward (macro)", True),
    ("reward_format_mean", "Format reward (macro)", True),
    ("reward_key_matching_mean", "Key-matching reward (macro)", True),
    ("reward_extraction_quality_positive_mean", "Extraction quality +(macro)", True),
    ("reward_extraction_quality_negative_mean", "Extraction quality -(macro)", True),
    ("reward_structured_total_mean", "Structured reward (macro)", True),
    ("json_errors_total", "JSON errors", False),
    ("n_truncated_total", "Truncated", False),
]

HEADER = ["Temp", "Top-p"] + [label for _, label, _ in METRIC_COLUMNS]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, required=True)
    parser.add_argument(
        "--pattern",
        type=str,
        default="grpo_metrics_*.json",
        help="Glob (relative to --metrics_dir) matching every metrics file to include.",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=None,
        help="Where to write the markdown table. Defaults to 'sweep_summary.md' inside --metrics_dir.",
    )
    parser.add_argument(
        "--pdf_out_path",
        type=Path,
        default=None,
        help="Where to write the PDF table. Defaults to 'sweep_summary.pdf' inside --metrics_dir.",
    )
    return parser.parse_args()


def sort_key(metrics):
    # A run is "greedy" when test_pipeline.py was called without --temperature/--top_p,
    # which leaves both fields null in the saved metrics; sort it before the grid.
    temperature, top_p = metrics.get("temperature"), metrics.get("top_p")
    if temperature is None and top_p is None:
        return (-1.0, -1.0)
    return (temperature, top_p)


def row_label(metrics):
    temperature, top_p = metrics.get("temperature"), metrics.get("top_p")
    if temperature is None and top_p is None:
        return "greedy", "-"
    return f"{temperature:.2f}", f"{top_p:.2f}"


def format_value(value):
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def build_cells(rows):
    """Return (header, cell_rows, bold_mask): cell_rows/bold_mask are lists of
    lists aligned with header, one row per run, shared by the markdown and PDF
    renderers so "what counts as the best value" is only computed once."""
    best = {}
    for key, _, higher_is_better in METRIC_COLUMNS:
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            best[key] = max(values) if higher_is_better else min(values)

    cell_rows, bold_mask = [], []
    for r in rows:
        cells = list(row_label(r))
        mask = [False, False]
        for key, _, _ in METRIC_COLUMNS:
            value = r.get(key)
            if value is None:
                cells.append("-")
                mask.append(False)
            else:
                cells.append(format_value(value))
                mask.append(key in best and value == best[key])
        cell_rows.append(cells)
        bold_mask.append(mask)
    return HEADER, cell_rows, bold_mask


def render_markdown(header, cell_rows, bold_mask):
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for cells, mask in zip(cell_rows, bold_mask):
        rendered = [f"**{c}**" if b else c for c, b in zip(cells, mask)]
        lines.append("| " + " | ".join(rendered) + " |")
    return "\n".join(lines)


def render_pdf(header, cell_rows, bold_mask, out_path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, landscape
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle

    doc = SimpleDocTemplate(str(out_path), pagesize=landscape(letter))
    table = Table([header] + cell_rows, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    for row_idx, mask in enumerate(bold_mask, start=1):
        for col_idx, is_bold in enumerate(mask):
            if is_bold:
                style.append(("FONTNAME", (col_idx, row_idx), (col_idx, row_idx), "Helvetica-Bold"))
    table.setStyle(TableStyle(style))
    doc.build([table])


def main():
    args = parse_args()
    paths = sorted(args.metrics_dir.glob(args.pattern))
    if not paths:
        raise SystemExit(f"No metrics files matching '{args.pattern}' found in {args.metrics_dir}")

    rows = []
    for path in paths:
        with open(path) as f:
            rows.append(json.load(f))
    rows.sort(key=sort_key)

    header, cell_rows, bold_mask = build_cells(rows)

    markdown_table = render_markdown(header, cell_rows, bold_mask)
    print(markdown_table)

    md_out_path = args.out_path or (args.metrics_dir / "sweep_summary.md")
    with open(md_out_path, "w") as f:
        f.write(markdown_table + "\n")
    print(f"\n[OK] Markdown table written to: {md_out_path}")

    pdf_out_path = args.pdf_out_path or (args.metrics_dir / "sweep_summary.pdf")
    render_pdf(header, cell_rows, bold_mask, pdf_out_path)
    print(f"[OK] PDF table written to: {pdf_out_path}")


if __name__ == "__main__":
    main()
