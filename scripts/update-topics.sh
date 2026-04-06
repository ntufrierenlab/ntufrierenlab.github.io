#!/usr/bin/env bash
#
# Update the topics field in a paper's frontmatter.
#
# Usage:
#   ./scripts/update-topics.sh <paper-filename> <comma-separated-topics>
#
# Example:
#   ./scripts/update-topics.sh my-paper.md "ISP,Agentic Pipeline"
#
set -euo pipefail

FILENAME="$1"
TOPICS="$2"
FILE="content/papers/${FILENAME}"

if [ ! -f "$FILE" ]; then
  echo "File not found: $FILE" >&2
  exit 1
fi

echo "Updating topics in ${FILE}" >&2

# Use environment variables to pass data safely to Python (no shell interpolation)
PAPER_FILE="$FILE" TOPICS_CSV="$TOPICS" python3 << 'PYEOF'
import os, re

filepath = os.environ['PAPER_FILE']
topics_csv = os.environ['TOPICS_CSV']

# Build YAML array from comma-separated string, sanitizing quotes
topics = [t.strip().replace('"', '').replace("'", '') for t in topics_csv.split(',') if t.strip()]
if not topics:
    topics = ['General']

topics_yaml = '[' + ','.join(f'"{t}"' for t in topics) + ']'

with open(filepath, 'r') as f:
    content = f.read()

content = re.sub(r'^topics:.*$', f'topics: {topics_yaml}', content, count=1, flags=re.MULTILINE)

with open(filepath, 'w') as f:
    f.write(content)

print(f'Topics set to: {topics_yaml}', flush=True)
PYEOF

echo "Done." >&2
