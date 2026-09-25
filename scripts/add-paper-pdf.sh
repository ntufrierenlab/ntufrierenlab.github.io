#!/usr/bin/env bash
#
# Add a paper to the knowledge base from a local PDF file.
# Uses Claude's pdf-extraction skill followed by paper-reading skill.
#
# Usage:
#   ./scripts/add-paper-pdf.sh <path-to-pdf> [topic-name]
#
# Example:
#   ./scripts/add-paper-pdf.sh ~/Downloads/my-paper.pdf "Color Constancy"
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CONTENT_DIR="${REPO_ROOT}/content/papers"

PDF_PATH="${1:?Usage: add-paper-pdf.sh <path-to-pdf> [topic-name]}"
TOPIC_NAME="${2:-General}"

if [ ! -f "$PDF_PATH" ]; then
    echo "Error: File not found: $PDF_PATH" >&2
    exit 1
fi

echo "Processing PDF: $PDF_PATH"
echo "Topic: $TOPIC_NAME"
echo ""

# Step 1: Extract structured content from PDF using Claude
echo "Step 1/2: Extracting content from PDF..."
EXTRACTION=$(claude --print "Use the pdf-extraction skill. Read and extract structured content from this paper PDF: $PDF_PATH" < "$PDF_PATH" 2>/dev/null)

if [ -z "$EXTRACTION" ]; then
    echo "Error: Failed to extract PDF content" >&2
    exit 1
fi

# Step 2: Generate bilingual summary using Claude
echo "Step 2/2: Generating bilingual summary..."

PROMPT="Use the paper-reading skill. Based on this extracted paper content, generate a complete Hugo markdown file (with YAML front matter and bilingual body content).

Topic: ${TOPIC_NAME}

${EXTRACTION}

Output ONLY the complete markdown file content, starting with --- for the front matter."

SUMMARY=$(echo "$PROMPT" | claude --print 2>/dev/null)

if [ -z "$SUMMARY" ]; then
    echo "Error: Failed to generate summary" >&2
    exit 1
fi

# Extract metadata for filename
DATE=$(echo "$SUMMARY" | grep '^date:' | head -1 | awk '{print $2}')
TITLE=$(echo "$SUMMARY" | grep '^title:' | head -1 | sed 's/^title: "//;s/"$//')

if [ -z "$DATE" ]; then
    DATE=$(date +%Y-%m-%d)
fi

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
