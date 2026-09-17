#!/usr/bin/env python3
"""Patch Qwen3.8-Flash-Next chat_template.jinja to MERGE multiple leading
system messages into one system block (instead of raising
'System message must be at the beginning.').

Hermes (and similar agents) send [system, system, ...] payloads; the stock
Qwen4Exp template accepts only a single messages[0] system. Same class of fix
as the DFlash2-fork template patch for the 27B lane.

Usage: python3 patch_chat_template.py /path/to/chat_template.jinja
"""
import shutil
import sys

T = sys.argv[1]
src = open(T).read()
orig = src

# ---- 1) insert merge pre-pass between the reasoning block and the tools branch ----
anchor = "{%- endif %}\n{%- if tools and tools is iterable and tools is not mapping %}"
merge_pass = (
    "{%- set ns_sysmerge = namespace(parts=[], leading=true) %}\n"
    "{%- for message in messages %}\n"
    "    {%- if ns_sysmerge.leading and message.role == 'system' %}\n"
    "        {%- set sys_content = render_content(message.content, false, true)|trim %}\n"
    "        {%- if sys_content %}\n"
    "            {%- set ns_sysmerge.parts = ns_sysmerge.parts + [sys_content] %}\n"
    "        {%- endif %}\n"
    "    {%- else %}\n"
    "        {%- set ns_sysmerge.leading = false %}\n"
    "    {%- endif %}\n"
    "{%- endfor %}\n"
    "{%- set merged_system = ns_sysmerge.parts | join('\\n\\n') %}\n"
)
assert src.count(anchor) == 1, f"anchor1 count {src.count(anchor)}"
src = src.replace(anchor, "{%- endif %}\n" + merge_pass + "{%- if tools and tools is iterable and tools is not mapping %}", 1)

# ---- 2) tools branch: use merged content ----
old2 = (
    "    {%- if messages[0].role == 'system' %}\n"
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
    "        {%- if content %}\n"
    "            {{- '\\n\\n' + content }}\n"
    "        {%- endif %}\n"
    "    {%- endif %}\n"
)
new2 = (
    "    {%- if merged_system %}\n"
    "        {{- '\\n\\n' + merged_system }}\n"
    "    {%- endif %}\n"
)
assert src.count(old2) == 1, f"anchor2 count {src.count(old2)}"
src = src.replace(old2, new2, 1)

# ---- 3) no-tools branch: use merged content ----
old3 = (
    "    {%- if messages[0].role == 'system' %}\n"
    "        {%- set content = render_content(messages[0].content, false, true)|trim %}\n"
    "        {%- if content %}\n"
    "            {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n' }}\n"
    "        {%- elif reasoning_instructions %}\n"
    "            {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n"
    "        {%- endif %}\n"
    "    {%- elif reasoning_instructions %}\n"
    "        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
)
new3 = (
    "    {%- if merged_system %}\n"
    "        {{- '<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + merged_system + '<|im_end|>\\n' }}\n"
    "    {%- elif reasoning_instructions %}\n"
    "        {{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}\n"
    "    {%- endif %}\n"
)
assert src.count(old3) == 1, f"anchor3 count {src.count(old3)}"
src = src.replace(old3, new3, 1)

# ---- 4) main loop: render nothing for system (merged above) ----
old4 = (
    '    {%- if message.role == "system" %}\n'
    '        {%- if not loop.first %}\n'
    "            {{- raise_exception('System message must be at the beginning.') }}\n"
    '        {%- endif %}\n'
    '    {%- elif message.role == "user" %}'
)
new4 = (
    '    {%- if message.role == "system" %}\n'
    '        {# system content merged into the leading block above #}\n'
    '    {%- elif message.role == "user" %}'
)
assert src.count(old4) == 1, f"anchor4 count {src.count(old4)}"
src = src.replace(old4, new4, 1)

if src == orig:
    print("no changes (already patched?)")
    sys.exit(0)

shutil.copy(T, T + ".orig-20260916")
open(T, "w").write(src)
print(f"[ok] patched {T} (backup {T}.orig-20260916)")
print("checks:")
print("  merged_system block present:", "merged_system" in src)
print("  raise removed:", "System message must be at the beginning" not in src)
