"""HTML + JSON report generation.

Two outputs:
  - report.json: machine-readable, for downstream tooling
  - report.html: self-contained, judges can double-click to open

The HTML report has:
  - Big number per detector (the bias score)
  - 4-bar chart (Chart.js via CDN)
  - Per-detector expandable section with examples
  - LLM-generated recommendation
  - Trajectory log (collapsible)
  - Debiased dataset download (if produced)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..schema import AnalysisReport


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>predec &mdash; {dataset_name} bias report</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    :root {{
      --bg: #0f1115;
      --card: #181b22;
      --border: #262a33;
      --text: #e6e8ec;
      --muted: #8a92a0;
      --accent: #6ee7b7;
      --warn: #fbbf24;
      --bad: #f87171;
      --ok: #34d399;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, -apple-system, "Segoe UI", sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    .container {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
    h1 {{ font-size: 28px; margin: 0 0 4px; }}
    h2 {{ font-size: 18px; margin: 32px 0 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    h3 {{ font-size: 15px; margin: 16px 0 8px; }}
    .sub {{ color: var(--muted); margin: 0 0 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 24px; }}
    .card {{
      background: var(--card); border: 1px solid var(--border); border-radius: 10px;
      padding: 18px;
    }}
    .card .name {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }}
    .card .score {{ font-size: 32px; font-weight: 700; margin: 6px 0 0; }}
    .card .ci {{ font-size: 11px; color: var(--muted); margin-top: 4px; }}
    .card.flagged {{ border-color: var(--bad); }}
    .card.flagged .score {{ color: var(--bad); }}
    .card.ok {{ border-color: var(--ok); }}
    .card.ok .score {{ color: var(--ok); }}
    .chart-wrap {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 24px; }}
    .bar-row {{ display: grid; grid-template-columns: 110px 1fr 60px; align-items: center; gap: 12px; margin: 10px 0; }}
    .bar-row .label {{ color: var(--muted); font-size: 13px; }}
    .bar-row .bar-track {{ position: relative; height: 18px; background: #11141a; border-radius: 9px; overflow: hidden; }}
    .bar-row .bar-fill {{ position: absolute; left: 0; top: 0; bottom: 0; }}
    .bar-row .bar-threshold {{ position: absolute; top: -2px; bottom: -2px; width: 2px; background: #8a92a0; }}
    .bar-row .bar-value {{ text-align: right; font-variant-numeric: tabular-nums; font-size: 13px; }}
    .detector {{ background: var(--card); border: 1px solid var(--border); border-radius: 10px; padding: 18px; margin-bottom: 16px; }}
    .detector .head {{ display: flex; justify-content: space-between; align-items: baseline; gap: 16px; }}
    .detector .head h3 {{ margin: 0; }}
    .badge {{ display: inline-block; font-size: 11px; padding: 2px 8px; border-radius: 999px; }}
    .badge.flagged {{ background: rgba(248,113,113,0.15); color: var(--bad); }}
    .badge.ok {{ background: rgba(52,211,153,0.15); color: var(--ok); }}
    .detector .metric {{ color: var(--muted); font-size: 13px; margin: 8px 0 16px; }}
    .example {{ background: #11141a; border: 1px solid var(--border); border-radius: 6px; padding: 10px 12px; margin-bottom: 8px; font-size: 13px; }}
    .example code {{ color: var(--accent); }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--muted); font-size: 13px; }}
    .recommendation {{ background: var(--card); border: 1px solid var(--border); border-left: 3px solid var(--accent); border-radius: 10px; padding: 20px 24px; margin-bottom: 24px; }}
    .recommendation p {{ margin: 0 0 8px; }}
    .meta {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--muted); font-weight: 500; text-transform: uppercase; font-size: 11px; letter-spacing: 0.05em; }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <div class="container">
    <h1>predec &mdash; bias report</h1>
    <p class="sub">dataset: <code>{dataset_name}</code> &middot; {n_pairs} pairs &middot; source: {dataset_source}</p>

    <div class="grid">
      {cards}
    </div>

    <div class="chart-wrap">
      <h3 style="margin-top:0;color:var(--muted);text-transform:uppercase;letter-spacing:0.05em;font-size:12px;">Bias scores vs flag thresholds</h3>
      {bar_rows}
    </div>

    <h2>Per-detector breakdown</h2>
    {detector_sections}

    <h2>Recommendation</h2>
    <div class="recommendation">
      <p>{recommendation}</p>
    </div>

    {debiased_section}

    <h2>Agent trajectories</h2>
    <details>
      <summary>Show {n_trajectory} trajectory events</summary>
      <div style="margin-top:12px;">
        <table>
          <thead><tr><th>Actor</th><th>Action</th><th>Decision</th><th>Duration (ms)</th></tr></thead>
          <tbody>{trajectory_rows}</tbody>
        </table>
      </div>
    </details>
  </div>
</body>
</html>
"""


def _format_score(det, threshold: float) -> tuple[str, str]:
    """Return (card_class, badge_html) for a detector result."""
    flagged = det.score >= threshold
    return ("flagged" if flagged else "ok"), (
        '<span class="badge flagged">flagged</span>' if flagged else '<span class="badge ok">within tolerance</span>'
    )


def _ci_str(ci: tuple[float, float]) -> str:
    if ci == (0.0, 0.0):
        return ""
    return f"95% CI [{ci[0]:.2f}, {ci[1]:.2f}]"


def render_html(report: AnalysisReport, debiased_path: str | None = None) -> str:
    """Render the full report as a self-contained HTML string."""
    ds = report.dataset
    dets = report.detectors

    # Top cards
    cards = []
    for name in ("length", "position", "sycophancy", "verbosity"):
        if name not in dets:
            continue
        d = dets[name]
        cls, _ = _format_score(d, d.threshold)
        ci = _ci_str(d.confidence_interval)
        cards.append(
            f'<div class="card {cls}">'
            f'<div class="name">{name}</div>'
            f'<div class="score">{d.score:.2f}</div>'
            f'<div class="ci">{ci}</div>'
            f"</div>"
        )

    # Chart data: build the bar chart rows (self-contained, no JS deps)
    bar_rows_parts = []
    for name, d in dets.items():
        flagged = d.score >= d.threshold
        color = "#f87171" if flagged else "#34d399"
        score_pct = float(d.score) * 100
        thresh_pct = float(d.threshold) * 100
        bar_rows_parts.append(
            f'<div class="bar-row">'
            f'<div class="label">{name}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{score_pct:.1f}%;background:{color};"></div>'
            f'<div class="bar-threshold" style="left:{thresh_pct:.1f}%;" title="threshold {d.threshold:.2f}"></div>'
            f'</div>'
            f'<div class="bar-value">{d.score:.2f}</div>'
            f'</div>'
        )
    bar_rows = "\n".join(bar_rows_parts)

    # Per-detector sections
    sections = []
    for name, d in dets.items():
        cls, badge = _format_score(d, d.threshold)
        examples_html = ""
        for ex in d.examples[:5]:
            evidence = ", ".join(f"{k}={v!r}" for k, v in ex.evidence.items())
            examples_html += (
                f'<div class="example">'
                f'<div><code>{ex.pair_id}</code></div>'
                f'<div>{ex.explanation}</div>'
                f'<div class="meta">{evidence}</div>'
                f"</div>"
            )
        if not examples_html:
            examples_html = '<div class="meta">no examples flagged</div>'

        extra_html = ""
        for k, v in (d.extra or {}).items():
            extra_html += f'<div class="meta">{k}: {v}</div>'

        sections.append(
            f'<div class="detector">'
            f'<div class="head"><h3>{d.name}</h3>{badge}</div>'
            f'<div class="metric">{d.metric_description}</div>'
            f"{examples_html}"
            f"{extra_html}"
            f"</div>"
        )

    # Debiased section
    if debiased_path:
        debiased_section = (
            '<h2>Debiased dataset</h2>'
            f'<p>A debiased version of the dataset is available at '
            f'<code>{debiased_path}</code> for downstream training.</p>'
        )
    else:
        debiased_section = ""

    # Trajectory rows
    traj = report.trajectory_log
    rows = []
    for ev in traj:
        rows.append(
            f"<tr><td>{ev['actor']}</td><td>{ev['action']}</td>"
            f"<td>{ev.get('decision', '')}</td>"
            f"<td>{ev.get('duration_ms', 0):.1f}</td></tr>"
        )

    return _HTML_TEMPLATE.format(
        dataset_name=ds.name,
        dataset_source=ds.source,
        n_pairs=ds.n_pairs,
        cards="\n".join(cards),
        bar_rows=bar_rows,
        detector_sections="\n".join(sections),
        recommendation=report.overall_recommendation,
        debiased_section=debiased_section,
        n_trajectory=len(traj),
        trajectory_rows="\n".join(rows),
    )


def render_json(report: AnalysisReport) -> str:
    return json.dumps(report.to_dict(), indent=2, default=str)


def write_report(
    report: AnalysisReport,
    out_dir: str,
    debiased_path: str | None = None,
) -> tuple[str, str]:
    """Write report.html and report.json to out_dir. Returns both paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    html = render_html(report, debiased_path=debiased_path)
    js = render_json(report)
    html_path = out / "report.html"
    json_path = out / "report.json"
    html_path.write_text(html)
    json_path.write_text(js)
    return str(html_path), str(json_path)
