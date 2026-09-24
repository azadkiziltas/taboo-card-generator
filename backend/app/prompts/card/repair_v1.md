---
name: card.repair
version: v1
language_strategy: english_instructions_native_output
created_at: 2026-09-24
description: Fix cards that failed deterministic validation while keeping the same target.
---
The following cards failed validation. Fix every card so that all errors are resolved.

Rules for fixing:
- Keep each "target" exactly the same.
- Replace only the problematic forbidden words; keep good ones.
- Each card needs exactly 5 different forbidden words in {{ language_name }} (ISO code: {{ language_code }}).
- Difficulty: {{ difficulty }}.

Cards and their validation errors:
{{ failures_json }}

Return exactly this JSON shape with the fixed cards:
{"cards": [{"target": "...", "forbidden_words": ["...", "...", "...", "...", "..."], "category": null, "difficulty": "{{ difficulty }}"}]}
