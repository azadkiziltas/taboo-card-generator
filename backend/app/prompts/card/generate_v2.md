---
name: card.generate
version: v2
language_strategy: english_instructions_native_output
created_at: 2026-09-24
description: Generate a batch of new cards for a category/difficulty.
---
Create {{ count }} new, distinct cards.

Output language: {{ language_name }} (ISO code: {{ language_code }})
Category: {{ category }}
Difficulty: {{ difficulty }}

Difficulty guide:
- easy: very common, concrete everyday words that children know.
- medium: well-known words and concepts that most adults know.
- hard: less frequent or more abstract words and concepts; still fair to describe.

<user_topic>{{ topic }}</user_topic>

Do not use any of these targets (they already exist): {{ avoid_targets }}

<reference_context>
{{ reference_context }}
</reference_context>

Return exactly this JSON shape, with {{ count }} items in "cards":
{"cards": [{"target": "...", "forbidden_words": ["...", "...", "...", "...", "..."], "category": "{{ category }}", "difficulty": "{{ difficulty }}"}]}
