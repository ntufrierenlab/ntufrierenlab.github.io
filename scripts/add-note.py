#!/usr/bin/env python3
"""Append a note to a paper's YAML front matter."""
import sys, os, datetime, re

filepath = sys.argv[1]
note_text = os.environ['NOTE_TEXT']

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

SEP = '---'
match = re.match(r'^' + SEP + r'\n(.*?\n)' + SEP + r'\n(.*)$', content, re.DOTALL)
if not match:
    print("Error: Could not parse front matter", file=sys.stderr)
    sys.exit(1)

fm_text = match.group(1)
body = match.group(2)

now = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
note_text_escaped = note_text.replace('\\', '\\\\').replace('"', '\\"')
note_entry = f'  - text: "{note_text_escaped}"\n    date: "{now}"'

if re.search(r'^notes:', fm_text, re.MULTILINE):
    lines = fm_text.split('\n')
    result = []
    in_notes = False
    inserted = False
    for i, line in enumerate(lines):
        if line.startswith('notes:'):
            in_notes = True
            result.append(line)
            continue
        if in_notes:
            if line.startswith('  -') or line.startswith('    '):
                result.append(line)
                continue
            else:
                result.append(note_entry)
                in_notes = False
                inserted = True
        result.append(line)
    if in_notes and not inserted:
        result.append(note_entry)
    fm_text = '\n'.join(result)
else:
    fm_text = fm_text.rstrip('\n') + '\nnotes:\n' + note_entry + '\n'

output = SEP + '\n' + fm_text + SEP + '\n' + body

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(output)

print(f"Note added successfully to {filepath}")
