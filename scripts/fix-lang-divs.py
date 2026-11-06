#!/usr/bin/env python3
"""Fix duplicate lang-en / lang-zh divs in paper summary files."""
import re, sys

filepath = sys.argv[1]
text = open(filepath).read()
parts = re.split(r'</div>\s*<div class="lang-en">\s*', text)
text = parts[0] + ''.join(parts[1:])
parts = re.split(r'</div>\s*<div class="lang-zh"[^>]*>\s*', text)
text = parts[0] + ''.join(parts[1:])
open(filepath, 'w').write(text)
print('Fixed: merged duplicate lang divs')
