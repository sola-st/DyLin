from fire import Fire
from pathlib import Path

def summarize_findings(results: str):
    results = Path(results)
    for result in results.glob('**/findings.csv'):
        