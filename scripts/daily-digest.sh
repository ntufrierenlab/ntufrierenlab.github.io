#!/usr/bin/env bash
#
# Daily Digest: fetch latest arXiv papers in Computer Vision, Image Processing,
# Computational Photography, and AI. Generate short bilingual summaries via Claude.
#
# Output: data/latest_digest.json
#
set -uo pipefail

MAX_PAPERS=10

# ── Fetch papers from arXiv API ──────────────────────────────────
# Categories: cs.CV (Computer Vision), eess.IV (Image and Video Processing), cs.AI (AI)
# Sort by submitted date descending, get recent papers

echo "Fetching latest papers from arXiv..." >&2

ARXIV_QUERY="cat:cs.CV+OR+cat:eess.IV+OR+cat:cs.AI"
ARXIV_URL="https://export.arxiv.org/api/query?search_query=${ARXIV_QUERY}&sortBy=submittedDate&sortOrder=descending&max_results=40"

ARXIV_XML=$(curl -sL "$ARXIV_URL")
if [ -z "$ARXIV_XML" ]; then
    echo "Error: Empty response from arXiv API" >&2
    exit 1
fi

# Parse XML and filter for relevant papers, output JSON
PAPERS_JSON=$(echo "$ARXIV_XML" | python3 -c "
import sys, xml.etree.ElementTree as ET, re, json

data = sys.stdin.read()
ns = {'atom': 'http://www.w3.org/2005/Atom', 'arxiv': 'http://arxiv.org/schemas/atom'}
root = ET.fromstring(data)
entries = root.findall('atom:entry', ns)

# Keywords to filter for relevance (4 core areas)
keywords = [
    # computer vision
    'computer vision', 'object detection', 'segmentation', 'depth estimation',
    'optical flow', 'visual understanding', 'scene understanding',
    'vision transformer', 'vit', '3d reconstruction', 'nerf',
    'gaussian splatting', 'point cloud', 'video understanding',
    # image processing
    'image processing', 'image restoration', 'image enhancement',
    'super-resolution', 'denoising', 'deblurring', 'inpainting',
    'hdr', 'high dynamic range', 'tone mapping', 'image compression',
    'image quality', 'image editing', 'video processing',
    # computational photography
    'computational photography', 'color constancy', 'white balance',
    'low-light', 'isp', 'image signal processing', 'raw',
    'demosaicing', 'retouching', 'aesthetic', 'neural rendering',
    'image synthesis', 'image generation', 'image-to-image',
    # artificial intelligence
    'artificial intelligence', 'deep learning', 'neural network',
    'transformer', 'diffusion model', 'generative model',
    'reinforcement learning', 'representation learning',
    'self-supervised', 'foundation model', 'large language model',
    'vision language', 'multimodal', 'contrastive learning',
]

papers = []
for entry in entries:
    title = re.sub(r'\s+', ' ', entry.find('atom:title', ns).text.strip())
    abstract = re.sub(r'\s+', ' ', entry.find('atom:summary', ns).text.strip())
    published = entry.find('atom:published', ns).text.strip()[:10]

    # Get arXiv ID from the entry id URL
    entry_id = entry.find('atom:id', ns).text.strip()
    arxiv_id = entry_id.split('/abs/')[-1]

    # Get categories
    categories = [c.attrib.get('term', '') for c in entry.findall('atom:category', ns)]
    primary_cat = categories[0] if categories else 'cs.CV'

    # Check relevance
    text_lower = (title + ' ' + abstract).lower()
    relevant = any(kw in text_lower for kw in keywords)

    if relevant and len(papers) < ${MAX_PAPERS}:
        # Get authors
        authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]

        papers.append({
            'title': title,
            'abstract': abstract,
            'authors': authors,
            'date': published,
            'arxiv_id': arxiv_id,
            'url': f'https://arxiv.org/abs/{arxiv_id}',
            'category': primary_cat,
        })

print(json.dumps(papers))
")

PAPER_COUNT=$(echo "$PAPERS_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")
echo "Found ${PAPER_COUNT} relevant papers" >&2

if [ "$PAPER_COUNT" -eq 0 ]; then
    echo "No relevant papers found, skipping digest" >&2
    exit 0
fi

# ── Build Claude prompt ──────────────────────────────────────────

PROMPT="You are a research assistant. For each paper below, write a concise speed-read summary:
- 2-3 sentences in English capturing the key contribution and result
- 2-3 sentences in Traditional Chinese (繁體中文) saying the same thing

Output ONLY valid JSON array. Each element: {\"title\": \"...\", \"summary\": \"English summary\", \"summary_zh\": \"中文摘要\"}

Keep the same order as input. Be specific about methods and results. No markdown, no code blocks, just raw JSON.

Papers:
$(echo "$PAPERS_JSON" | python3 -c "
import sys, json
papers = json.load(sys.stdin)
for i, p in enumerate(papers):
    print(f\"\\n--- Paper {i+1} ---\")
    print(f\"Title: {p['title']}\")
    print(f\"Abstract: {p['abstract'][:500]}\")
")"

# ── Call Claude ──────────────────────────────────────────────────

echo "Calling Claude for summaries..." >&2
SUMMARIES=$(echo "$PROMPT" | claude --print --model claude-haiku-4-5-20251001 2>/dev/null)

if [ -z "$SUMMARIES" ]; then
    echo "Error: Claude returned empty response" >&2
    exit 1
fi
echo "Claude response received (${#SUMMARIES} chars)" >&2

# ── Merge summaries with paper metadata into final JSON ──────────

TODAY=$(date +%Y-%m-%d)

python3 -c "
import sys, json, re

papers_raw = json.loads('''$PAPERS_JSON''')
summaries_raw = '''$(echo "$SUMMARIES" | sed "s/'/'\\''/g")'''

# Extract JSON array from Claude response (strip any markdown fences)
summaries_raw = re.sub(r'^[^\[]*', '', summaries_raw, count=1)
summaries_raw = re.sub(r'[^\]]*$', '', summaries_raw, count=1)

try:
    summaries = json.loads(summaries_raw)
except json.JSONDecodeError:
    print('Error: Failed to parse Claude JSON response', file=sys.stderr)
    print(summaries_raw[:500], file=sys.stderr)
    sys.exit(1)

result = {
    'date': '$TODAY',
    'papers': []
}

for i, paper in enumerate(papers_raw):
    entry = {
        'title': paper['title'],
        'url': paper['url'],
        'category': paper['category'],
        'authors': paper['authors'][:3],
    }
    if i < len(summaries):
        entry['summary'] = summaries[i].get('summary', '')
        entry['summary_zh'] = summaries[i].get('summary_zh', '')
    else:
        entry['summary'] = ''
        entry['summary_zh'] = ''
    result['papers'].append(entry)

print(json.dumps(result, indent=2, ensure_ascii=False))
" > /tmp/latest_digest.json

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate digest JSON" >&2
    exit 1
fi

# ── Save to data directory ───────────────────────────────────────

mkdir -p data
cp /tmp/latest_digest.json data/latest_digest.json
echo "Saved digest to data/latest_digest.json ($(wc -c < data/latest_digest.json) bytes)" >&2
