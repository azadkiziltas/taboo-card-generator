---
name: card.generate_for_targets
version: v1
language_strategy: english_instructions_native_output
created_at: 2026-09-24
description: Generate forbidden words for given target words (single card, regeneration, benchmark).
---
Create one card for each of these target words. Keep every target exactly as written.

Targets: {{ targets_json }}

Output language: {{ language_name }} (ISO code: {{ language_code }})
Category: {{ category }}
Difficulty: {{ difficulty }}

Avoid these previously used forbidden words if possible: {{ avoid_forbidden }}

<user_topic>{{ topic }}</user_topic>

<reference_context>
{{ reference_context }}
</reference_context>

Return exactly this JSON shape, one item per target:
{"cards": [{"target": "...", "forbidden_words": ["...", "...", "...", "...", "..."], "category": "{{ category }}", "difficulty": "{{ difficulty }}"}]}
