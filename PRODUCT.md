# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

- Game client: Flutter (planned, not yet in this repository).
- Backend: Python 3.12, FastAPI, PostgreSQL + pgvector, Ollama (`backend/`).
- Internal usage UI: plain static HTML/CSS/JS with no build step, served by the FastAPI backend from the same origin (user's choice).

## Users

- **Primary for the internal UI:** the developer/team building the game. They generate cards through the backend, judge card quality by eye, and record feedback. They use it at a desk while iterating on prompts, models, and validation rules.
- **End users of the product:** players of a multilingual word-description party game (Taboo-style), through the future Flutter app. They never choose a model or provider.

## Product Purpose

A multilingual word-guessing party game. The backend generates cards: a target word plus exactly 5 forbidden words, in the requested language, category, and difficulty. It uses LLMs behind a provider-agnostic router, validates every card deterministically, and stores only valid cards.

The internal UI exists so the team can exercise that pipeline and evaluate the output without curl or Swagger. Success means the team can quickly produce cards, see whether they are good, and leave `good/bad` and difficulty feedback that feeds future benchmarks and fine-tuning data.

## Positioning

Cards are generated natively per language (TR→TR, EN→EN, …), never by translating. Every card passes the same deterministic rules (5 unique forbidden words, target not among them, script checks, duplicate checks) before anyone sees it. The model behind generation can change without the product changing.

## Operating Context

- Generation is synchronous: with the local default model a 5-card request takes about 20 s (roughly 5 s plus 4 s per card), and larger requests take longer. The UI must make that wait legible.
- Supported languages: `tr`, `en`, `de`, `fr`, `es`, `ar`. Arabic is right-to-left; the backend stores Unicode as-is and presentation is the client's job.
- Difficulty is optional: `easy | medium | hard`, or none. With none, nothing about difficulty is sent to the model and the card has no difficulty level. The category is free text: a short label (`technology`, `food`) or a theme (`uzay yolculuğu`). The internal UI has one field for both. The public API still accepts a separate optional `topic`.
- Public API used by the UI: `POST /v1/cards/generate`, `POST /v1/cards/{id}/feedback` (rating `good|bad`, difficulty_feedback `too_easy|correct|too_hard`), `POST /v1/cards/{id}/regenerate`, `GET /v1/health`.
- Errors come back as `{"error": {"code", "message"}}` with codes such as `generation_unavailable`, `generation_timeout`, `generation_failed`, `invalid_request`.

## Capabilities and Constraints

- In scope for the internal UI (confirmed): generating cards and reviewing them, per-card feedback, regenerating a card.
- Explicitly out of scope for now: play mode, admin panel (registry, telemetry, RAG), listing existing cards from the database.
- The UI must never let the user pick a model or provider; the API rejects such fields.
- Partial results are possible: fewer cards than requested may come back.

## Evidence on Hand

- Real generated cards exist in the local database (e.g. TR technology: "Yapay Zeka", forbidden words Algoritma, Bilgisayar, Öğrenme, Robot, Simülasyon).
- No logo, brand name, brand voice, or visual identity exists yet. Do not invent one as fact.

## Product Principles

1. The model is invisible to players and the UI; quality is judged on the card, not the model.
2. Only validated cards are ever shown or stored.
3. Every language is first-class, including right-to-left scripts.
4. Feedback is cheap to give, because it is the training and benchmark signal.
