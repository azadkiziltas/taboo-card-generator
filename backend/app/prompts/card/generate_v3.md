---
name: card.generate
version: v3
language_strategy: english_instructions_native_output
created_at: 2026-09-24
description: Generate a batch of new cards for a category or theme; difficulty is optional.
---
Create {{ count }} new, distinct cards.

Output language: {{ language_name }} (ISO code: {{ language_code }})
Category or theme: {{ category }}
{{ difficulty_section }}

<user_topic>{{ topic }}</user_topic>

Do not use any of these targets (they already exist): {{ avoid_targets }}

<reference_context>
{{ reference_context }}
</reference_context>

Return exactly this JSON shape, with {{ count }} items in "cards":
{"cards": [{"target": "...", "forbidden_words": ["...", "...", "...", "...", "..."], "category": "{{ category }}"}]}
