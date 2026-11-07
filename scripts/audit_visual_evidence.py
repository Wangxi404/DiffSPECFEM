#!/usr/bin/env python3
"""Run the installed scientific-figure source and final-PDF QA tools."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--qa-scripts', type=Path, default=Path(os.environ.get(
    'NATURE_FIGURE_QA', Path.home()/'.codex/skills/nature-figure/scripts')))
args = p.parse_args()
qa = ROOT/'figures/qa'
qa.mkdir(exist_ok=True)
checks = [('source-validation', [str(args.qa_scripts/'validate_figure.py'),
           str(ROOT/'scripts/visualize_evidence.py'), '--backend', 'python', '--json'])]
for pdf in sorted((ROOT/'figures').glob('*.pdf')):
    checks.extend([
        (pdf.stem+'.text', [str(args.qa_scripts/'audit_pdf_text.py'), str(pdf), '--json']),
        (pdf.stem+'.collision', [str(args.qa_scripts/'audit_figure_collisions.py'), str(pdf),
            '--json-out', str(qa/(pdf.stem+'.collision.json')),
            '--overlay-pdf', str(qa/(pdf.stem+'.collision.pdf'))])])

def run(item):
    name, command = item
    if name.endswith('.collision'):
        # The auditor creates overlays only for findings; remove a stale one.
        (qa/(name+'.pdf')).unlink(missing_ok=True)
    result = subprocess.run([sys.executable, *command], text=True, capture_output=True)
    # Strip workstation-specific prefixes from QA logs before committing them.
    out = result.stdout.replace(str(ROOT)+'/', '').replace(str(args.qa_scripts)+'/', '')
    (qa/(name+'.log')).write_text(out+result.stderr)
    if name.endswith('.collision'):
        artifact = qa/(name+'.json')
        if artifact.exists():
            artifact.write_text(artifact.read_text().replace(str(ROOT)+'/', ''))
    return {'check': name, 'exit_code': result.returncode, 'log': 'figures/qa/'+name+'.log'}

with ThreadPoolExecutor(max_workers=2) as pool:
    report = list(pool.map(run, checks))
(qa/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
sys.exit(int(any(row['exit_code'] for row in report)))
