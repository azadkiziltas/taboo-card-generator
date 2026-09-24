---
version: 1
slug: "backend-app-ui-static-index-html"
primary_target: "backend/app/ui/static/index.html"
related_targets: ["backend/app/ui/static/app.css","backend/app/ui/static/app.js"]
---

# Surface: internal card generation & review UI

Scope: `backend/app/ui/static/` (index.html, app.css, app.js), served by FastAPI at `/ui/`. Mode: Operate.

Audience/job: the dev team generates cards via `POST /v1/cards/generate`, reviews each card, sends good/bad + difficulty feedback, and regenerates weak cards. Desk use; generation takes 25–70 s.

Constraints: static HTML/CSS/JS, no build step; UI copy in Turkish; no model/provider controls; Arabic cards render RTL; cards shown in original casing (no text-transform). Out of scope: play mode, admin panel, DB card listing.

Chosen: the category standard (canon), user-picked; quality bar: GitHub (Primer).

## Direction contract

THESIS: A canon internal tool at GitHub's craft level: request form beside a reviewable card list, and no metaphor. Refuses chart dashboards, hero banners, and card-grid decoration.
OWN-WORLD: Primer-style neutrals (white canvas, #f6f8fa panels, 1px #d1d9e0 borders, 6px radius), system UI stack, blue #0969da for links and focus, green primary button, and semantic labels always paired with an icon and text. Dark theme follows the OS.
STORY: Pick language, an optional difficulty (default: none), and count, and type a category or theme into the single category field; generate; watch an honest elapsed timer; review; mark good/bad and difficulty; regenerate weak cards.
FIRST VIEWPORT: Header with the title and a health indicator. A 320px left Box form ends in a full-width green "Kart üret" button. On the right: a batch header with counts and a filter segmented control, then the card rows: large target, 5 numbered forbidden words, meta labels, and an action row. Signature: GitHub-style keyboard shortcuts (j/k, g/b, 1/2/3, r, ?).
FORM: canon, standing exit taken by the user; seed 89a8381b.
FINISH: unreviewed and undocumented is unfinished; this build ends with the finish review, the verdict, DESIGN.md, and every shipping raster carrying its provenance
