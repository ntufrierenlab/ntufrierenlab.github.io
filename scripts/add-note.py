#!/usr/bin/env python3
"""Add or delete a note in a paper's YAML front matter."""
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

# Check if this is a delete operation
if note_text.startswith('__DELETE__:'):
    target_date = note_text[len('__DELETE__:'):]
    # Remove the note with the matching date
    lines = fm_text.split('\n')
    result = []
    in_notes = False
    skip_entry = False
    for line in lines:
        if line.startswith('notes:'):
            in_notes = True
            result.append(line)
            continue
        if in_notes:
            if line.startswith('  - '):
                # Start of a note entry — check if next line has matching date
                skip_entry = False
                # Check if this line's date matches
                if f'date: "{target_date}"' in line:
                    skip_entry = True
                    continue
                # Check paired date on this entry
                result.append(line)
                continue
            if line.startswith('    '):
                # Continuation of a note entry (e.g. date line)
                if skip_entry:
                    continue
                if f'date: "{target_date}"' in line:
                    # Remove the previous text line too
                    if result and result[-1].startswith('  - text:'):
                        result.pop()
                    skip_entry = True
                    continue
                result.append(line)
                continue
            else:
                in_notes = False
                skip_entry = False
        result.append(line)

    fm_text = '\n'.join(result)

    # If notes array is now empty, remove the notes: key
    has_note_entries = any(line.strip().startswith('- text:') for line in result)
    if not has_note_entries:
        result = [line for line in result if line.rstrip() != 'notes:']
        fm_text = '\n'.join(result)

    print(f"Note deleted from {filepath}")
else:
    # Add operation
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

    print(f"Note added to {filepath}")

output = SEP + '\n' + fm_text + SEP + '\n' + body

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(output)
