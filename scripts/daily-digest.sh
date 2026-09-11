#!/usr/bin/env bash
#
# Daily Digest: fetch top papers from Hugging Face Daily Papers.
# Uses HF's built-in AI summaries — no Claude needed.
#
# Output: data/latest_digest.json
#
set -uo pipefail

MAX_PAPERS=5
TODAY=$(date +%Y-%m-%d)

# ── Fetch papers from Hugging Face Daily Papers API ──────────────

# Try today, then go back up to 3 days to find the latest date with papers
HF_JSON=""
for OFFSET in 0 1 2 3; do
    TRY_DATE=$(date -d "-${OFFSET} days" +%Y-%m-%d 2>/dev/null || date -v-${OFFSET}d +%Y-%m-%d)
    echo "Trying ${TRY_DATE}..." >&2
    HF_JSON=$(curl -sL "https://huggingface.co/api/daily_papers?date=${TRY_DATE}")
    PAPER_COUNT=$(echo "$HF_JSON" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))" 2>/dev/null || echo "0")
    if [ "$PAPER_COUNT" -gt 0 ]; then
        TODAY="$TRY_DATE"
        echo "Found ${PAPER_COUNT} papers for ${TODAY}" >&2
        break
    fi
done

if [ "$PAPER_COUNT" -eq 0 ]; then
    echo "No papers found in the last 4 days, skipping" >&2
    exit 0
fi

# ── Parse and select top papers by upvotes ───────────────────────

echo "$HF_JSON" > /tmp/hf_papers.json

export MAX_PAPERS TODAY
python3 << 'PYEOF' > /tmp/latest_digest.json
import json, sys, os

with open('/tmp/hf_papers.json') as f:
    raw = json.load(f)

MAX_PAPERS = int(os.environ.get('MAX_PAPERS', '10'))
TODAY = os.environ.get('TODAY', '')

papers = sorted(raw, key=lambda x: x.get('paper', {}).get('upvotes', 0), reverse=True)[:MAX_PAPERS]

result = {
    'date': TODAY,
    'papers': []
}

for item in papers:
    p = item.get('paper', {})
    authors = [a.get('name', '') for a in p.get('authors', []) if not a.get('hidden', False)]

    result['papers'].append({
        'title': p.get('title', ''),
        'url': f"https://huggingface.co/papers/{p.get('id', '')}",
        'arxiv_url': f"https://arxiv.org/abs/{p.get('id', '')}",
        'summary': p.get('ai_summary', ''),
        'upvotes': p.get('upvotes', 0),
        'authors': authors[:3],
    })

json.dump(result, sys.stdout, indent=2, ensure_ascii=False)
PYEOF

if [ $? -ne 0 ]; then
    echo "Error: Failed to generate digest JSON" >&2
    exit 1
fi

# ── Save to data directory ───────────────────────────────────────

mkdir -p data
cp /tmp/latest_digest.json data/latest_digest.json
echo "Saved digest ($(python3 -c "import json; print(len(json.load(open('/tmp/latest_digest.json'))['papers']))" ) papers) to data/latest_digest.json" >&2
