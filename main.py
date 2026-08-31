import warnings
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys
import time
from pathlib import Path

from src.hallucination_supervisor.pipeline import SupervisorPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def run_evaluation(prompt: str):
    console.print(Panel(f"[bold cyan]Input Prompt:[/bold cyan]\n{prompt}", title="Task Started"))

    # Initialize the pipeline
    pipeline = SupervisorPipeline()

    with console.status("[bold green]Agent Pipeline Running... (This may take 10-30 seconds)[/bold green]"):
        start_time = time.time()
        try:
            # Execute the pipeline — use the detailed entry point so we get
            # claims + verdicts back, not just the aggregated report.
            result = pipeline.run_verification_detailed(prompt)
        except Exception as e:
            console.print(f"[bold red]Pipeline Failed:[/bold red] {e}")
            return

    execution_time = time.time() - start_time
    report = result.report  # the aggregated VerificationReport still lives here

    # Print the Original Draft
    console.print("\n[bold yellow]Base Generator Draft:[/bold yellow]")
    console.print(result.draft)
    print("\n" + "="*50 + "\n")

    # Build the Verification Table
    table = Table(title=f"Multi-Agent Verification Results ({execution_time:.1f}s)", show_header=True, header_style="bold magenta")
    table.add_column("Claim ID", style="dim", width=10)
    table.add_column("Extracted Factual Claim", width=40)
    table.add_column("Verdict", justify="center", width=15)
    table.add_column("Confidence", justify="right")
    table.add_column("Judge Rationale")

    if result.claims:
        for claim in result.claims:
            verdict = result.verdict_for(claim.claim_id)
            if verdict is None:
                continue  # defensive: shouldn't happen, but don't crash the table on it

            status_color = "green" if verdict.status == "SUPPORTED" else "red" if verdict.status == "CONTRADICTED" else "yellow"

            table.add_row(
                verdict.claim_id[:8],          # ClaimVerdict has claim_id, not claim_text
                claim.text,                     # claim text comes from the AtomicClaim, not the verdict
                f"[bold {status_color}]{verdict.status}[/bold {status_color}]",
                f"{verdict.confidence_score * 100:.0f}%",
                verdict.rationale
            )
    else:
        table.add_row("—", "[i]No verifiable claims extracted from this draft.[/i]", "—", "—", "—")

    console.print(table)

    # Print Final Score
    score_color = "green" if report.hallucination_percentage < 20 else "red"
    console.print(f"\n[bold]Final Hallucination Score:[/bold] [{score_color}]{report.hallucination_percentage:.1f}%[/{score_color}]")

if __name__ == "__main__":

    test_prompt = "Who is the current prime minister of India?"
    run_evaluation(test_prompt)