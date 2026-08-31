"""predec CLI (typer).

Subcommands:
  detect    - Run the bias detectors on a dataset, write report
  debias    - Apply a debiasing strategy, write a new JSONL
  eval      - Run the agentic vs baseline comparison
  version   - Print the predec version
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from .debiaser import Strategy, debias, to_jsonl
from .llm.client import LLMClient
from .loaders import load_preference_dataset
from .orchestrator import run_analysis
from .report.render import write_report

app = typer.Typer(
    name="predec",
    help="Detect and quantify biases in RLHF preference datasets.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def detect(
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="HuggingFace dataset name (e.g. anthropic/hh-rlhf)"),
    split: str = typer.Option("train", "--split", help="HF split"),
    input: Optional[str] = typer.Option(None, "--input", "-i", help="Path to a local JSONL/CSV/Parquet file"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Cap the number of pairs analyzed"),
    out: str = typer.Option("runs/detect", "--out", "-o", help="Output directory for the report"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Skip the sycophancy detector (no LLM call)"),
):
    """Run all 4 bias detectors and write report.html + report.json."""
    if dataset:
        source = {"hf": dataset, "split": split, "limit": limit}
        name = dataset
        source_str = f"huggingface:{dataset}"
    elif input:
        path = Path(input)
        if path.suffix == ".jsonl":
            source = {"jsonl": str(path), "limit": limit}
        elif path.suffix == ".csv":
            source = {"csv": str(path), "limit": limit}
        elif path.suffix in (".parquet", ".pq"):
            source = {"parquet": str(path), "limit": limit}
        else:
            typer.echo(f"Unsupported file format: {path.suffix}", err=True)
            raise typer.Exit(code=1)
        name = path.stem
        source_str = f"file:{path}"
    else:
        typer.echo("Provide either --dataset or --input", err=True)
        raise typer.Exit(code=1)

    console.print(f"[bold]predec[/bold] loading {source_str} (limit={limit})")
    pairs = load_preference_dataset(source)
    console.print(f"  loaded {len(pairs)} pairs")

    llm = None if no_llm else LLMClient()
    if llm and not llm.available:
        console.print("[yellow]OPENAI_API_KEY not set; sycophancy detector will be skipped[/yellow]")
        llm = None

    report = run_analysis(pairs, dataset_name=name, dataset_source=source_str, llm=llm)

    # Save trajectory separately
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "trajectory.json").write_text(json.dumps(report.trajectory_log, indent=2, default=str))

    html_path, json_path = write_report(report, str(out_path))
    console.print(f"[green]wrote {html_path}[/green]")
    console.print(f"[green]wrote {json_path}[/green]")

    # Print summary table
    table = Table(title=f"predec results: {name}")
    table.add_column("Detector", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Status")
    for name_, d in report.detectors.items():
        status = "[red]flagged[/red]" if d.score >= d.flag_threshold else "[green]ok[/green]"
        table.add_row(name_, f"{d.score:.3f}", status)
    table.add_row("[bold]overall[/bold]", f"{report.overall_bias_score:.3f}", "")
    console.print(table)
    console.print(f"\nLLM cost: ${sum(c.cost_usd for c in llm.calls):.4f}" if llm else "")


@app.command()
def debias_cmd(
    input: str = typer.Option(..., "--input", "-i", help="Path to input JSONL preference dataset"),
    report: str = typer.Option(..., "--report", "-r", help="Path to a predec report.json"),
    strategy: Strategy = typer.Option("reweight", "--strategy", "-s", help="Debiasing strategy"),
    out: str = typer.Option(..., "--out", "-o", help="Output path for the debiased JSONL"),
):
    """Apply a debiasing strategy using a prior report."""
    from .schema import DetectorResult

    pairs = load_preference_dataset({"jsonl": input})
    with open(report) as f:
        report_data = json.load(f)

    # Reconstruct DetectorResult dict
    results = {}
    for name, d in report_data["detectors"].items():
        results[name] = DetectorResult(
            name=d["name"],
            score=d["score"],
            confidence_interval=tuple(d["confidence_interval"]),
            n_pairs_analyzed=d["n_pairs_analyzed"],
            n_pairs_flagged=d["n_pairs_flagged"],
            metric_description=d["metric_description"],
            extra=d.get("extra", {}),
        )

    res = debias(pairs, results, strategy=strategy)
    to_jsonl(res.pairs, res.weights, path=out)
    console.print(f"[green]wrote {len(res.pairs)} pairs to {out}[/green]")
    for note in res.notes:
        console.print(f"  - {note}")


@app.command()
def eval(
    out: str = typer.Option("runs/eval", "--out", "-o", help="Output directory"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Cap the eval set size"),
):
    """Run the agentic pipeline vs single-prompt baseline and report F1."""
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent.parent / "scripts" / "run_eval.py"),
        "--out", out,
    ]
    if limit:
        cmd.extend(["--limit", str(limit)])
    raise typer.Exit(subprocess.call(cmd))


@app.command()
def version():
    """Print the predec version."""
    from . import __version__
    console.print(f"predec {__version__}")


if __name__ == "__main__":
    app()
