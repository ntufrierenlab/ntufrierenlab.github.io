#!/usr/bin/env bash
#
# Summarize an arXiv paper using Claude CLI.
# Generates a bilingual (EN/ZH) Hugo markdown file.
#
# Usage:
#   ./scripts/summarize.sh <arxiv-url> [topic-name]
#
# Example:
#   ./scripts/summarize.sh https://arxiv.org/abs/2004.01354 "Auto White Balance"
#
set -uo pipefail

ARXIV_URL="${1:?Usage: summarize.sh <arxiv-url> [topic-name]}"
TOPIC_NAME="${2:-General}"

# Extract arXiv ID
ARXIV_ID=$(echo "$ARXIV_URL" | grep -oE '[0-9]{4}\.[0-9]+' | head -1)
if [ -z "$ARXIV_ID" ]; then
    echo "Error: Could not extract arXiv ID from URL" >&2
    exit 1
fi

echo "Processing arXiv ID: ${ARXIV_ID}" >&2
PDF_URL="https://arxiv.org/pdf/${ARXIV_ID}"

# Fetch paper abstract from arXiv API
echo "Fetching metadata from arXiv API..." >&2
ARXIV_XML=$(curl -sL "https://export.arxiv.org/api/query?id_list=${ARXIV_ID}")
if [ -z "$ARXIV_XML" ]; then
    echo "Error: Empty response from arXiv API" >&2
    exit 1
fi

ABSTRACT=$(echo "$ARXIV_XML" | python3 -c "
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

if echo "$ABSTRACT" | grep -q '^TITLE='; then
    echo "Metadata extracted successfully" >&2
else
    echo "Error: Failed to extract metadata" >&2
    echo "$ABSTRACT" >&2
    exit 1
fi

# Parse metadata
TITLE=$(echo "$ABSTRACT" | grep '^TITLE=' | cut -d= -f2-)
AUTHORS=$(echo "$ABSTRACT" | grep '^AUTHORS=' | cut -d= -f2-)
DATE=$(echo "$ABSTRACT" | grep '^DATE=' | cut -d= -f2-)
PAPER_ABSTRACT=$(echo "$ABSTRACT" | grep '^ABSTRACT=' | cut -d= -f2-)

# Format authors as YAML list
AUTHORS_YAML=$(echo "$AUTHORS" | tr ',' '\n' | sed 's/^ *//' | sed 's/^/  - "/' | sed 's/$/"/')

# Build the Claude prompt
PROMPT="You are a research paper analyst. Read this paper and generate a structured summary in BOTH English and Chinese.

Paper title: ${TITLE}
Paper abstract: ${PAPER_ABSTRACT}
PDF URL: ${PDF_URL}

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

# Call Claude
echo "Calling Claude API..." >&2
SUMMARY=$(echo "$PROMPT" | claude --print --model claude-sonnet-4-5-20250929 2>&2)
if [ -z "$SUMMARY" ]; then
    echo "Error: Claude returned empty response" >&2
    exit 1
fi
echo "Claude response received (${#SUMMARY} chars)" >&2

# Extract one-line summaries
ONE_LINE_EN=$(echo "$SUMMARY" | grep '^ONE_LINE_EN=' | cut -d= -f2- | sed 's/^"//;s/"$//')
ONE_LINE_ZH=$(echo "$SUMMARY" | grep '^ONE_LINE_ZH=' | cut -d= -f2- | sed 's/^"//;s/"$//')

# Remove the ONE_LINE markers from the body
BODY=$(echo "$SUMMARY" | grep -v '^ONE_LINE_EN=' | grep -v '^ONE_LINE_ZH=')

# Generate the full markdown file
cat <<FRONTMATTER
---
title: "${TITLE}"
date: ${DATE}
authors:
${AUTHORS_YAML}
arxiv_url: "${ARXIV_URL}"
pdf_url: "${PDF_URL}"
one_line_summary: "${ONE_LINE_EN}"
one_line_summary_zh: "${ONE_LINE_ZH}"
topics: ["${TOPIC_NAME}"]
tags: []
---

${BODY}
FRONTMATTER
