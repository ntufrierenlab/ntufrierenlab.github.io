#!/usr/bin/env bash
#
# Summarize a preprint paper (arXiv or bioRxiv) using Claude CLI.
# Generates a bilingual (EN/ZH) Hugo markdown file.
#
# Usage:
#   ./scripts/summarize.sh <paper-url> [topic-name] [doi] [pdf-url]
#
# Examples:
#   ./scripts/summarize.sh https://arxiv.org/abs/2004.01354 "Auto White Balance"
#   ./scripts/summarize.sh https://www.biorxiv.org/content/10.1101/2026.01.22.700600v1 "Topic"
#   ./scripts/summarize.sh "" "Topic" "10.1109/CVPR.2024.12345" "https://openaccess.thecvf.com/..."
#
set -uo pipefail

PAPER_URL="${1:-}"
TOPIC_NAME="${2:-General}"
DOI="${3:-}"
EXT_PDF_URL="${4:-}"

if [ -z "$PAPER_URL" ] && [ -z "$DOI" ]; then
    echo "Usage: summarize.sh <paper-url> [topic-name] [doi] [pdf-url]" >&2
    echo "Either paper-url or doi must be provided." >&2
    exit 1
fi

# ── Detect source and extract metadata ─────────────────────────────

if echo "$PAPER_URL" | grep -qi 'biorxiv\.org'; then
    SOURCE="bioRxiv"
    echo "Detected bioRxiv paper" >&2

    # Fetch the HTML page and extract metadata from <meta> tags
    echo "Fetching metadata from bioRxiv page..." >&2
    PAGE_HTML=$(curl -sL "$PAPER_URL")

    METADATA=$(echo "$PAGE_HTML" | python3 -c "
import sys, re, html as htmlmod
data = sys.stdin.read()

def meta(name):
    # Match <meta name=\"...\" content=\"...\"> or <meta property=\"...\" content=\"...\">
    m = re.search(r'<meta\s+(?:name|property)=\"' + re.escape(name) + r'\"\s+content=\"([^\"]*?)\"', data)
    if not m:
        m = re.search(r'<meta\s+content=\"([^\"]*?)\"\s+(?:name|property)=\"' + re.escape(name) + r'\"', data)
    return htmlmod.unescape(m.group(1).strip()) if m else ''

title = meta('DC.Title') or meta('citation_title') or meta('og:title') or ''
date = meta('DC.Date') or meta('citation_publication_date') or meta('article:published_time') or ''
doi = meta('citation_doi') or ''
abstract = meta('DC.Description') or meta('og:description') or ''

# Extract authors — use citation_author only to avoid duplicates with DC.Contributor
authors = re.findall(r'<meta\s+(?:name=\"citation_author\"\s+content=\"([^\"]*?)\"|content=\"([^\"]*?)\"\s+name=\"citation_author\")', data)
author_list = [a[0] or a[1] for a in authors if (a[0] or a[1]).strip()]
# Deduplicate while preserving order
seen = set()
unique_authors = []
for a in author_list:
    if a not in seen:
        seen.add(a)
        unique_authors.append(a)
author_list = unique_authors

if date and '/' in date:
    # Convert MM/DD/YYYY or YYYY/MM/DD to YYYY-MM-DD
    parts = date.split('/')
    if len(parts[0]) == 4:
        date = f'{parts[0]}-{parts[1].zfill(2)}-{parts[2].zfill(2)}'
    else:
        date = f'{parts[2]}-{parts[0].zfill(2)}-{parts[1].zfill(2)}'

if not title:
    print('ERROR=No title found in page', file=sys.stderr)
    sys.exit(1)

print(f'TITLE={title}')
print(f'AUTHORS={chr(44).join(author_list)}')
print(f'DATE={date[:10]}')
print(f'ABSTRACT={abstract}')
print(f'DOI={doi}')
" 2>&1)

    # Extract DOI for PDF URL construction
    PAGE_DOI=$(echo "$METADATA" | grep '^DOI=' | cut -d= -f2-)
    if [ -n "$PAGE_DOI" ]; then
        PDF_URL="https://www.biorxiv.org/content/${PAGE_DOI}v1.full.pdf"
    else
        # Fallback: try to get PDF URL from the page
        PDF_URL=$(echo "$PAGE_HTML" | grep -oE 'https://www\.biorxiv\.org/content/[^"]+\.full\.pdf' | head -1)
        if [ -z "$PDF_URL" ]; then
            echo "Error: Could not determine PDF URL" >&2
            exit 1
        fi
    fi
    echo "PDF URL: ${PDF_URL}" >&2

elif [ -n "$PAPER_URL" ]; then
    SOURCE="arXiv"
    # Extract arXiv ID
    ARXIV_ID=$(echo "$PAPER_URL" | grep -oE '[0-9]{4}\.[0-9]+' | head -1)
    if [ -z "$ARXIV_ID" ]; then
        echo "Error: Could not extract arXiv ID from URL" >&2
        exit 1
    fi

    echo "Processing arXiv ID: ${ARXIV_ID}" >&2
    PDF_URL="https://arxiv.org/pdf/${ARXIV_ID}"

    # Fetch metadata from arXiv API
    echo "Fetching metadata from arXiv API..." >&2
    ARXIV_XML=$(curl -sL "https://export.arxiv.org/api/query?id_list=${ARXIV_ID}")
    if [ -z "$ARXIV_XML" ]; then
        echo "Error: Empty response from arXiv API" >&2
        exit 1
    fi

    METADATA=$(echo "$ARXIV_XML" | python3 -c "
import sys, xml.etree.ElementTree as ET, re
data = sys.stdin.read()
ns = {'atom': 'http://www.w3.org/2005/Atom'}
root = ET.fromstring(data)
entry = root.find('atom:entry', ns)
if entry is not None:
    title = re.sub(r'\s+', ' ', entry.find('atom:title', ns).text.strip())
    summary = re.sub(r'\s+', ' ', entry.find('atom:summary', ns).text.strip())
    authors = [a.find('atom:name', ns).text.strip() for a in entry.findall('atom:author', ns)]
    published = entry.find('atom:published', ns).text.strip()[:10]
    print(f'TITLE={title}')
    print(f'AUTHORS={chr(44).join(authors)}')
    print(f'DATE={published}')
    print(f'ABSTRACT={summary}')
else:
    print('ERROR=No entry found', file=sys.stderr)
    sys.exit(1)
" 2>&1)

elif [ -n "$DOI" ]; then
    # ── Conference paper path: fetch metadata from OpenAlex via DOI ──
    echo "Detected conference paper with DOI: ${DOI}" >&2
    echo "Fetching metadata from OpenAlex API..." >&2

    OA_JSON=$(curl -sL "https://api.openalex.org/works/doi:${DOI}?mailto=ntufrierenlab@gmail.com")
    if [ -z "$OA_JSON" ] || echo "$OA_JSON" | grep -q '"error"'; then
        echo "Error: OpenAlex API returned error or empty response" >&2
        echo "$OA_JSON" >&2
        exit 1
    fi

    METADATA=$(echo "$OA_JSON" | python3 -c "
import sys, json, re

data = json.load(sys.stdin)

title = (data.get('title') or 'Untitled').strip()
authors = [a['author']['display_name'] for a in data.get('authorships', []) if a.get('author', {}).get('display_name')]
pub_date = (data.get('publication_date') or '')[:10]

# Reconstruct abstract from inverted index
inv = data.get('abstract_inverted_index') or {}
if inv:
    words = {}
    for word, positions in inv.items():
        for pos in positions:
            words[pos] = word
    abstract = ' '.join(words[i] for i in sorted(words.keys()))
else:
    abstract = ''

# Identify venue/source name
source_name = ''
pl = data.get('primary_location') or {}
src = pl.get('source') or {}
source_name = src.get('display_name') or ''

# Find PDF URL from best_oa_location or locations
pdf_url = ''
boa = data.get('best_oa_location') or {}
pdf_url = boa.get('pdf_url') or ''
if not pdf_url:
    for loc in data.get('locations', []):
        if loc.get('pdf_url'):
            pdf_url = loc['pdf_url']
            break

# Landing page URL
landing_url = pl.get('landing_page_url') or ''
if not landing_url:
    landing_url = f'https://doi.org/{data.get(\"doi\", \"\").replace(\"https://doi.org/\", \"\")}'

print(f'TITLE={title}')
print(f'AUTHORS={chr(44).join(authors)}')
print(f'DATE={pub_date}')
print(f'ABSTRACT={abstract}')
print(f'SOURCE_NAME={source_name}')
print(f'OA_PDF_URL={pdf_url}')
print(f'LANDING_URL={landing_url}')
" 2>&1)

    # Determine source label from venue name
    OA_SOURCE_NAME=$(echo "$METADATA" | grep '^SOURCE_NAME=' | cut -d= -f2-)
    OA_PDF_URL=$(echo "$METADATA" | grep '^OA_PDF_URL=' | cut -d= -f2-)
    OA_LANDING_URL=$(echo "$METADATA" | grep '^LANDING_URL=' | cut -d= -f2-)

    # Map venue name to short label
    SOURCE=$(echo "$OA_SOURCE_NAME" | python3 -c "
import sys, re
name = sys.stdin.read().strip().lower()
mappings = [
    (r'cvpr|computer vision and pattern recognition', 'CVPR'),
    (r'\biccv\b|international conference on computer vision(?! and pattern)', 'ICCV'),
    (r'\beccv\b|european conference on computer vision', 'ECCV'),
    (r'\bwacv\b|winter conference on applications of computer vision', 'WACV'),
    (r'\baccv\b|asian conference on computer vision', 'ACCV'),
    (r'neurips|neural information processing', 'NeurIPS'),
    (r'\bicml\b|international conference on machine learning', 'ICML'),
    (r'\biclr\b|international conference on learning representations', 'ICLR'),
    (r'\baaai\b', 'AAAI'),
    (r'siggraph asia', 'SIGGRAPH Asia'),
    (r'siggraph', 'SIGGRAPH'),
    (r'\bijcv\b|international journal of computer vision', 'IJCV'),
    (r'\btpami\b|transactions on pattern analysis', 'TPAMI'),
]
for pattern, label in mappings:
    if re.search(pattern, name):
        print(label)
        sys.exit(0)
print('Paper')
")

    echo "Source: ${SOURCE} (${OA_SOURCE_NAME})" >&2

    # PDF URL priority: explicit argument > OpenAlex
    if [ -n "$EXT_PDF_URL" ]; then
        PDF_URL="$EXT_PDF_URL"
    elif [ -n "$OA_PDF_URL" ]; then
        PDF_URL="$OA_PDF_URL"
    else
        echo "Error: No PDF URL available for this paper" >&2
        exit 1
    fi

    # Paper URL (landing page) for front matter
    if [ -z "$PAPER_URL" ]; then
        PAPER_URL="$OA_LANDING_URL"
    fi

    echo "PDF URL: ${PDF_URL}" >&2
fi

# ── Validate metadata ──────────────────────────────────────────────

if echo "$METADATA" | grep -q '^TITLE='; then
    echo "Metadata extracted successfully" >&2
else
    echo "Error: Failed to extract metadata" >&2
    echo "$METADATA" >&2
    exit 1
fi

# Parse metadata
TITLE=$(echo "$METADATA" | grep '^TITLE=' | cut -d= -f2-)
AUTHORS=$(echo "$METADATA" | grep '^AUTHORS=' | cut -d= -f2-)
DATE=$(echo "$METADATA" | grep '^DATE=' | cut -d= -f2-)
PAPER_ABSTRACT=$(echo "$METADATA" | grep '^ABSTRACT=' | cut -d= -f2-)

# Format authors as YAML list
AUTHORS_YAML=$(echo "$AUTHORS" | tr ',' '\n' | sed 's/^ *//' | sed 's/^/  - "/' | sed 's/$/"/')

# ── Download PDF and extract text ──────────────────────────────────

echo "Downloading PDF from ${PDF_URL}..." >&2
curl -sL -o /tmp/paper.pdf "$PDF_URL"
PDF_SIZE=$(wc -c < /tmp/paper.pdf)
echo "PDF downloaded (${PDF_SIZE} bytes)" >&2

echo "Extracting text from PDF..." >&2
PAPER_TEXT=$(pdftotext /tmp/paper.pdf - 2>/dev/null | head -c 50000)
TEXT_LEN=${#PAPER_TEXT}
echo "Extracted ${TEXT_LEN} chars of text" >&2

if [ "$TEXT_LEN" -lt 500 ]; then
    echo "Warning: Very little text extracted from PDF, using abstract only" >&2
    PAPER_TEXT="$PAPER_ABSTRACT"
fi

# ── Build the Claude prompt ────────────────────────────────────────

PROMPT="You are a senior researcher and top-conference (NeurIPS/CVPR/ICLR/ICML) reviewer. Read this paper thoroughly and generate a structured, in-depth summary in BOTH English and Traditional Chinese.

Paper title: ${TITLE}
Paper abstract: ${PAPER_ABSTRACT}
PDF URL: ${PDF_URL}

--- FULL PAPER TEXT ---
${PAPER_TEXT}
--- END OF PAPER TEXT ---

You have been given the full paper text above. Use the actual content, data, methods, and results from the paper to generate a detailed and accurate analysis.

Please generate the summary content (NOT the front matter, just the body) in this exact format:

<div class=\"lang-en\">

## Key Contributions

Write 4-6 detailed bullet points about the paper's main contributions. For each contribution:
- Start with a **bolded short label** summarizing it, then explain in 2-3 sentences
- Describe WHAT the contribution is, WHY it matters, and HOW it differs from prior work
- Include specific technical details: what method/framework/model is proposed, what problem it solves
- Mention the key novelty compared to existing approaches
- If the paper introduces a dataset or benchmark, describe its scale and characteristics

## Core Insights

Write 4-6 detailed bullet points about the key insights and findings. For each insight:
- Explain the underlying reasoning or intuition behind why the approach works
- Reference specific experiments, ablations, or theoretical analysis that support the insight
- Include concrete numbers (accuracy, speedup, etc.) when available
- Connect insights to broader implications for the field
- Highlight any surprising or counter-intuitive findings

## Key Data & Results

[Include a markdown table comparing the proposed method with baselines on the main benchmarks. Use exact numbers from the paper. Include dataset names, metrics, and all compared methods.]

- Write 3-5 bullet points discussing key quantitative results, including:
  - Performance on main benchmarks vs. state-of-the-art
  - Ablation study results showing which components matter most
  - Computational cost, inference speed, or parameter count comparisons
  - Any failure cases or scenarios where the method underperforms

## Strengths

Analyze from the perspective of a top-conference reviewer (NeurIPS/CVPR/ICLR). Write 4-6 bullet points:
- Evaluate the novelty and significance of the technical contribution
- Assess the experimental design: Are baselines comprehensive and fair? Are datasets appropriate?
- Judge the clarity of writing, motivation, and presentation
- Comment on theoretical grounding or mathematical rigor if applicable
- Evaluate reproducibility: Are implementation details sufficient?
- Note if the paper opens new research directions or provides useful tools/datasets to the community

## Weaknesses

Analyze from the perspective of a top-conference reviewer (NeurIPS/CVPR/ICLR). Write 4-6 bullet points:
- Identify missing baselines or comparisons that should have been included
- Point out experimental gaps: missing datasets, limited evaluation metrics, or insufficient ablations
- Highlight unclear assumptions, unjustified design choices, or logical gaps
- Note scalability concerns, computational overhead, or deployment limitations
- Identify overclaimed results or areas where conclusions are not fully supported by evidence
- Mention potential negative societal impact if relevant

## Research Directions

Imagine you are a PhD student or postdoc wanting to build on this paper to produce a top-conference publication. Write 5-7 bullet points, each describing a concrete research direction:
- For each direction, explain: (1) the specific idea, (2) why it would be impactful, (3) a rough approach to implement it
- Consider: extending to new domains/modalities, addressing the identified weaknesses, combining with complementary techniques, scaling up, theoretical analysis
- Think about what would make a strong NeurIPS/CVPR/ICLR submission building on this work
- Prioritize directions that are both novel and feasible within 6-12 months of research

</div>

<div class=\"lang-zh\" style=\"display:none;\">

## 主要貢獻

[Translate the Key Contributions section into Traditional Chinese with the same level of detail]

## 核心洞見

[Translate the Core Insights section into Traditional Chinese with the same level of detail]

## 關鍵數據與結果

[Translate the Key Data & Results section including the table into Traditional Chinese]

## 優勢

[Translate the Strengths section into Traditional Chinese, maintaining the top-conference reviewer perspective]

## 劣勢

[Translate the Weaknesses section into Traditional Chinese, maintaining the top-conference reviewer perspective]

## 研究方向

[Translate the Research Directions section into Traditional Chinese, maintaining the PhD student/postdoc perspective]

</div>

Important:
- Use **bold** for key terms and method names
- Keep paper titles, method names, dataset names, and technical terms in English even in the Chinese version
- Use Traditional Chinese (繁體中文), not Simplified Chinese
- Be specific with numbers, data, and citations from the paper — do NOT fabricate results
- Each section should be thorough and substantive, not superficial
- CRITICAL: At the very end, output two one-line summaries on their own lines with NO markdown formatting (no bold, no quotes, no backticks):
ONE_LINE_EN=your english one-line summary here
ONE_LINE_ZH=your chinese one-line summary here"

# ── Call Claude ────────────────────────────────────────────────────

echo "Calling Claude API..." >&2
for ATTEMPT in 1 2 3; do
    SUMMARY=$(printf '%s' "$PROMPT" | claude --print --model claude-haiku-4-5-20251001 2>/dev/null)
    if [ -z "$SUMMARY" ]; then
        echo "Error: Claude returned empty response (attempt ${ATTEMPT})" >&2
        continue
    fi
    echo "Claude response received (${#SUMMARY} chars, attempt ${ATTEMPT})" >&2
    # Require at least 2000 chars and the lang-en div to consider it valid
    if [ "${#SUMMARY}" -ge 2000 ] && printf '%s' "$SUMMARY" | grep -q '<div class="lang-en">'; then
        break
    fi
    echo "Response too short or missing body content, retrying..." >&2
    sleep 2
done

if [ -z "$SUMMARY" ] || [ "${#SUMMARY}" -lt 2000 ]; then
    echo "Error: Claude failed to generate sufficient content after 3 attempts" >&2
    exit 1
fi

# Save raw response to temp file for robust Python-based extraction
printf '%s\n' "$SUMMARY" > /tmp/claude_response.txt
echo "Raw response saved ($(wc -c < /tmp/claude_response.txt) bytes)" >&2

# Use Python for robust extraction — avoids bash echo/grep/sed pipeline issues
# that can silently drop body content due to special characters or formatting
python3 << 'EXTRACT_EOF'
import sys

with open('/tmp/claude_response.txt', 'r') as f:
    text = f.read()

lines = text.split('\n')

one_line_en = ''
one_line_zh = ''
body_lines = []

for line in lines:
    stripped = line.strip().replace('*', '')
    upper = stripped.upper()
    if upper.startswith('ONE_LINE_EN') and ('=' in stripped or ':' in stripped):
        sep = '=' if '=' in stripped else ':'
        val = stripped.split(sep, 1)[1].strip().strip('"').strip("'")
        one_line_en = val
    elif upper.startswith('ONE_LINE_ZH') and ('=' in stripped or ':' in stripped):
        sep = '=' if '=' in stripped else ':'
        val = stripped.split(sep, 1)[1].strip().strip('"').strip("'")
        one_line_zh = val
    else:
        body_lines.append(line)

# Find body starting from <div class="lang-en">
body_text = '\n'.join(body_lines)
idx = body_text.find('<div class="lang-en">')
if idx >= 0:
    body = body_text[idx:]
else:
    # Fallback: use everything after removing ONE_LINE markers
    body = body_text

with open('/tmp/paper_one_line_en.txt', 'w') as f:
    f.write(one_line_en)
with open('/tmp/paper_one_line_zh.txt', 'w') as f:
    f.write(one_line_zh)
with open('/tmp/paper_body.txt', 'w') as f:
    f.write(body)

print(f"Extracted: ONE_LINE_EN={len(one_line_en)} chars, ONE_LINE_ZH={len(one_line_zh)} chars, BODY={len(body)} chars", file=sys.stderr)
EXTRACT_EOF

ONE_LINE_EN=$(cat /tmp/paper_one_line_en.txt)
ONE_LINE_ZH=$(cat /tmp/paper_one_line_zh.txt)

BODY_LEN=$(wc -c < /tmp/paper_body.txt | tr -d ' ')
echo "Body extracted: ${BODY_LEN} bytes" >&2

if [ "$BODY_LEN" -lt 1000 ]; then
    echo "Error: Body content too short (${BODY_LEN} bytes), expected at least 1000" >&2
    echo "First 500 chars of raw Claude response:" >&2
    head -c 500 /tmp/claude_response.txt >&2
    echo "" >&2
    exit 1
fi

# ── Generate the full markdown file ────────────────────────────────

TOPICS_YAML=$(echo "$TOPIC_NAME" | tr ',' '\n' | sed 's/^ *//;s/ *$//' | sed 's/.*/"&"/' | paste -sd',' - | sed 's/^/[/;s/$/]/')

# Write front matter (heredoc is fine here — no untrusted content)
cat <<FRONTMATTER
---
title: "${TITLE}"
date: ${DATE}
authors:
${AUTHORS_YAML}
source: "${SOURCE}"
arxiv_url: "${PAPER_URL}"
pdf_url: "${PDF_URL}"
one_line_summary: "${ONE_LINE_EN}"
one_line_summary_zh: "${ONE_LINE_ZH}"
date_added: $(date +%Y-%m-%d)
topics: ${TOPICS_YAML}
tags: []
---

FRONTMATTER
# Append body from file — avoids bash interpretation of $, `, \ in body content
cat /tmp/paper_body.txt
