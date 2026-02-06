#!/usr/bin/env bash
#
# Summarize a preprint paper (arXiv or bioRxiv) using Claude CLI.
# Generates a bilingual (EN/ZH) Hugo markdown file.
#
# Usage:
#   ./scripts/summarize.sh <paper-url> [topic-name]
#
# Examples:
#   ./scripts/summarize.sh https://arxiv.org/abs/2004.01354 "Auto White Balance"
#   ./scripts/summarize.sh https://www.biorxiv.org/content/10.1101/2026.01.22.700600v1 "Topic"
#
set -uo pipefail

PAPER_URL="${1:?Usage: summarize.sh <paper-url> [topic-name]}"
TOPIC_NAME="${2:-General}"

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

else
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
PAPER_TEXT=$(pdftotext /tmp/paper.pdf - 2>/dev/null | head -c 150000)
TEXT_LEN=${#PAPER_TEXT}
echo "Extracted ${TEXT_LEN} chars of text" >&2

if [ "$TEXT_LEN" -lt 500 ]; then
    echo "Warning: Very little text extracted from PDF, using abstract only" >&2
    PAPER_TEXT="$PAPER_ABSTRACT"
fi

# ── Build the Claude prompt ────────────────────────────────────────

PROMPT="You are a research paper analyst. Read this paper and generate a structured summary in BOTH English and Chinese.

Paper title: ${TITLE}
Paper abstract: ${PAPER_ABSTRACT}
PDF URL: ${PDF_URL}

--- FULL PAPER TEXT ---
${PAPER_TEXT}
--- END OF PAPER TEXT ---

You have been given the full paper text above. Use the actual content, data, and results from the paper to generate a detailed and accurate summary.

Please generate the summary content (NOT the front matter, just the body) in this exact format:

<div class=\"lang-en\">

## Key Contributions

- [3-5 bullet points about the paper's main contributions]

## Core Insights

- [3-4 bullet points about the key insights and findings]

## Key Data & Results

[Include a markdown table comparing results with baselines if applicable]

- [2-4 bullet points about key quantitative results]

## Strengths

- [3-4 bullet points about the paper's strengths]

## Weaknesses

- [3-4 bullet points about the paper's weaknesses]

## Potential Improvements

- [4-5 bullet points about possible improvements]

</div>

<div class=\"lang-zh\" style=\"display:none;\">

## 主要貢獻

- [same content in Traditional Chinese]

## 核心洞見

- [same content in Traditional Chinese]

## 關鍵數據與結果

[same table in Traditional Chinese]

- [same bullet points in Traditional Chinese]

## 優勢

- [same content in Traditional Chinese]

## 劣勢

- [same content in Traditional Chinese]

## 可改進方向

- [same content in Traditional Chinese]

</div>

Important:
- Use **bold** for key terms
- Keep paper titles, method names, and technical terms in English even in the Chinese version
- Use Traditional Chinese (繁體中文), not Simplified Chinese
- Be specific with numbers and data from the paper
- Also output two one-line summaries: one in English and one in Traditional Chinese, on separate lines prefixed with ONE_LINE_EN= and ONE_LINE_ZH="

# ── Call Claude ────────────────────────────────────────────────────

echo "Calling Claude API..." >&2
SUMMARY=$(echo "$PROMPT" | claude --print --model claude-haiku-4-5-20251001 2>&2)
if [ -z "$SUMMARY" ]; then
    echo "Error: Claude returned empty response" >&2
    exit 1
fi
echo "Claude response received (${#SUMMARY} chars)" >&2

# Extract one-line summaries and sanitize (remove quotes that break YAML)
ONE_LINE_EN=$(echo "$SUMMARY" | grep '^ONE_LINE_EN=' | cut -d= -f2- | sed 's/^"//;s/"$//;s/"//g')
ONE_LINE_ZH=$(echo "$SUMMARY" | grep '^ONE_LINE_ZH=' | cut -d= -f2- | sed 's/^"//;s/"$//;s/"//g')

# Remove the ONE_LINE markers and any Claude preamble before the first <div> from the body
BODY=$(echo "$SUMMARY" | grep -v '^ONE_LINE_EN=' | grep -v '^ONE_LINE_ZH=' | sed '1,/<div class="lang-en">/{/<div class="lang-en">/!d}')

# ── Generate the full markdown file ────────────────────────────────

cat <<FRONTMATTER
---
title: "${TITLE}"
date: ${DATE}
authors:
${AUTHORS_YAML}
arxiv_url: "${PAPER_URL}"
pdf_url: "${PDF_URL}"
one_line_summary: "${ONE_LINE_EN}"
one_line_summary_zh: "${ONE_LINE_ZH}"
topics: ["${TOPIC_NAME}"]
tags: []
---

${BODY}
FRONTMATTER
