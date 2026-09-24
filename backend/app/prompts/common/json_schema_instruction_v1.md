---
name: common.json_schema_instruction
version: v1
language_strategy: english_instructions
created_at: 2026-09-24
description: Appended to the system prompt for models without native schema enforcement.
---
Your entire answer must be one JSON object that validates against this JSON Schema. Do not wrap it in markdown code fences and do not add any text before or after it.

JSON Schema:
{{ schema }}
