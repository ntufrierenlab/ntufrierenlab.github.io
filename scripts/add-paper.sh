#!/usr/bin/env bash
#
# Manually add a paper to the knowledge base.
#
# Usage:
#   ./scripts/add-paper.sh <arxiv-url> [topic-name]
#
# Example:
#   ./scripts/add-paper.sh https://arxiv.org/abs/2004.01354 "Auto White Balance"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTENT_DIR="${REPO_ROOT}/content/papers"

ARXIV_URL="${1:?Usage: add-paper.sh <arxiv-url> [topic-name]}"
TOPIC_NAME="${2:-General}"

# Extract arXiv ID
ARXIV_ID=$(echo "$ARXIV_URL" | grep -oE '[0-9]{4}\.[0-9]+' | head -1)
if [ -z "$ARXIV_ID" ]; then
    echo "Error: Could not extract arXiv ID from URL" >&2
    exit 1
fi

# Check if paper already exists
SAFE_ID=$(echo "$ARXIV_ID" | tr '.' '-')
if ls "${CONTENT_DIR}"/*"${SAFE_ID}"* 2>/dev/null | head -1 > /dev/null 2>&1; then
    echo "Paper ${ARXIV_ID} already exists in the knowledge base."
    exit 0
fi

echo "Generating summary for ${ARXIV_URL}..."
echo "Topic: ${TOPIC_NAME}"
echo ""

# Generate summary
SUMMARY=$("${SCRIPT_DIR}/summarize.sh" "$ARXIV_URL" "$TOPIC_NAME")

if [ -z "$SUMMARY" ]; then
    echo "Error: Failed to generate summary" >&2
    exit 1
fi

# Extract date and title for filename
DATE=$(echo "$SUMMARY" | grep '^date:' | head -1 | awk '{print $2}')
TITLE=$(echo "$SUMMARY" | grep '^title:' | head -1 | sed 's/^title: "//;s/"$//')
SAFE_TITLE=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]/-/g' | sed 's/--*/-/g' | head -c 60)

FILENAME="${DATE}-${SAFE_TITLE}.md"
FILEPATH="${CONTENT_DIR}/${FILENAME}"

mkdir -p "$CONTENT_DIR"
echo "$SUMMARY" > "$FILEPATH"

echo ""
echo "Paper saved to: ${FILEPATH}"
echo ""
echo "To preview:"
echo "  hugo server -D"
echo ""
echo "To publish:"
echo "  git add '${FILEPATH}' && git commit -m 'Add paper: ${TITLE}'"
