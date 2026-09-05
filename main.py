import warnings

from Project.src.hallucination_supervisor.agents.judge import NLIVerifier
warnings.filterwarnings("ignore", category=ResourceWarning)

import sys
import time
from pathlib import Path

from src.hallucination_supervisor.agents import (
    BaseDraftGenerator,
    ClaimExtractor,
    EvidenceInvestigator
)

from src.hallucination_supervisor.pipeline import SupervisorPipeline
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

console = Console()

def run_evaluation(prompt: str):
    console.print(Panel(f"[bold cyan]Input Prompt:[/bold cyan]\n{prompt}", title="Task Started"))

    # Initialize the pipeline
    pipeline = SupervisorPipeline(generator=BaseDraftGenerator(provider="openai", model_name="gpt-4o"),
    extractor=ClaimExtractor(provider="ollama", model_name="llama3.1:8b"),
    investigator=EvidenceInvestigator(provider="ollama", model_name="llama3.1:8b"),
    judge=NLIVerifier(provider="ollama", model_name="gemma2:9b"),
)

    with console.status("[bold green]Agent Pipeline Running... (This may take 10-30 seconds)[/bold green]"):
        start_time = time.time()
        try:
            result = pipeline.run_verification_detailed(prompt)
        except Exception as e:
            console.print(f"[bold red]Pipeline Failed:[/bold red] {e}")
            return

    execution_time = time.time() - start_time
    report = result.report

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
                continue  # shouldn't happen

            status_color = "green" if verdict.status == "SUPPORTED" else "red" if verdict.status == "CONTRADICTED" else "yellow"

            table.add_row(
                verdict.claim_id[:8],          # ClaimVerdict has claim_id
                claim.text,                     # claim text comes from the AtomicClaim
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