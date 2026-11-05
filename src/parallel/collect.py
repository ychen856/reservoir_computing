"""Combine complete summary rows from finished runs, without selecting best cases."""
import argparse
import csv
import json
from pathlib import Path
from src.parallel.common import write_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('runs',type=Path,nargs='+',help='Properties/benchmarks output directories')
    p.add_argument('--output',type=Path,default=Path('results/parallel_all_results.csv'))
    a=p.parse_args()
    rows=[]
    for run in a.runs:
        config=json.loads((run/'config.json').read_text())
        with (run/'summary.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                rows.append(dict(source_run=str(run),smoke=config['smoke'],
                    amplitude=config['amplitude'],ridge=config['ridge'],k=config['k'],
                    requested_cycles_per_symbol=config['cycles_per_symbol'],
                    config_file=str(run/'config.json'),**row))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    write_csv(a.output,rows)
    print(f'Saved {len(rows)} rows: {a.output}')

if __name__=='__main__':
    main()
