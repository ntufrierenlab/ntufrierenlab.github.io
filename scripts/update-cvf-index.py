#!/usr/bin/env python3
"""Build CVF Open Access paper index for search."""
import urllib.request, re, json, sys, os

CONFERENCES = [
    'ICCV2025', 'CVPR2025', 'WACV2025',
    'CVPR2024', 'WACV2024',
    'ICCV2023', 'CVPR2023',
]

OUTPUT = os.path.join(os.path.dirname(__file__), '..', 'static', 'data', 'cvf-index.json')

all_papers = {}
for conf in CONFERENCES:
    url = f'https://openaccess.thecvf.com/{conf}?day=all'
    print(f'Fetching {conf}...', file=sys.stderr)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        html = urllib.request.urlopen(req, timeout=60).read().decode('utf-8', errors='replace')
        entries = re.findall(
            r'href="(/content/[^"]+_paper\.html)">\s*([^<]+?)\s*</a>', html)
        papers = []
        for path, title in entries:
            pdf = path.replace('/html/', '/papers/').replace('_paper.html', '_paper.pdf')
            papers.append({'t': title.strip(), 'p': pdf})
        all_papers[conf] = papers
        print(f'  {len(papers)} papers', file=sys.stderr)
    except Exception as e:
        print(f'  ERROR: {e}', file=sys.stderr)
        all_papers[conf] = []

os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
with open(OUTPUT, 'w') as f:
    json.dump(all_papers, f, ensure_ascii=False, separators=(',', ':'))

total = sum(len(v) for v in all_papers.values())
print(f'Written {total} papers to {OUTPUT}', file=sys.stderr)
