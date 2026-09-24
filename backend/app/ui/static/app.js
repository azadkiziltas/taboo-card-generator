// Kart Üretimi: internal UI for generating and reviewing cards.
// Talks only to the public API on the same origin; never picks a model or provider.

const API = {
  generate: "/v1/cards/generate",
  feedback: (id) => `/v1/cards/${encodeURIComponent(id)}/feedback`,
  regenerate: (id) => `/v1/cards/${encodeURIComponent(id)}/regenerate`,
  health: "/v1/health",
};

const LANGUAGE_NAMES = { tr: "Türkçe", en: "İngilizce", de: "Almanca", fr: "Fransızca", es: "İspanyolca", ar: "Arapça" };
const RTL_LANGUAGES = new Set(["ar"]);
const DIFFICULTY_NAMES = { easy: "Kolay", medium: "Orta", hard: "Zor" };
const DIFFICULTY_FEEDBACK = { too_easy: "Kolay", correct: "Uygun", too_hard: "Zor" };
const RATING_NAMES = { good: "İyi", bad: "Kötü" };
// Benchmarked with the local default model (M4 Pro): ~5 s overhead plus ~4 s per card.
const expectedSeconds = (count) => Math.round(5 + 4 * count);
const STORE_KEY = "taboo-card-ui:v1";
const HEALTH_INTERVAL_MS = 30_000;

const ERROR_MESSAGES = {
  generation_unavailable: ["Model şu anda erişilemiyor", "Ollama'nın çalıştığını kontrol edip tekrar dene."],
  generation_busy: ["Model meşgul", "Birkaç saniye sonra tekrar dene."],
  generation_timeout: ["Model zamanında yanıt vermedi", "Daha az kart isteyerek tekrar dene."],
  generation_failed: ["Geçerli kart üretilemedi", "Model doğrulamadan geçen kart üretemedi. Tekrar dene ya da kategoriyi değiştir."],
  storage_unavailable: ["Veritabanına ulaşılamıyor", "PostgreSQL'in çalıştığını kontrol et."],
  context_unavailable: ["Referans verisine ulaşılamıyor", "RAG servisi yanıt vermiyor."],
  rate_limited: ["Çok fazla istek", "Biraz bekleyip tekrar dene."],
  card_not_found: ["Kart bulunamadı", "Kart veritabanında yok ya da yenisiyle değiştirilmiş."],
  invalid_request: ["İstek geçersiz", "Form alanlarını kontrol et."],
  network: ["Sunucuya ulaşılamadı", "Backend çalışıyor mu? `docker compose up -d` ile başlatabilirsin."],
  internal_error: ["Sunucu hatası", "Beklenmeyen bir hata oluştu; backend loglarına bak."],
};

class ApiError extends Error {
  constructor(code, title, text, details = []) {
    super(title);
    this.code = code;
    this.title = title;
    this.text = text;
    this.details = details;
  }
}

// ---------------------------------------------------------------------------------------
// state

const state = {
  batches: [], // newest first: { id, createdAt, request, elapsedMs, cards: [card] }
  filter: "all",
  running: null, // { startedAt, request, controller, interval }
  lastRequest: null,
  selectedId: null,
};

function loadState() {
  try {
    const raw = localStorage.getItem(STORE_KEY);
    if (!raw) return;
    const saved = JSON.parse(raw);
    if (Array.isArray(saved.batches)) state.batches = saved.batches;
    if (saved.form) applyForm(saved.form);
  } catch {
    /* storage unavailable or corrupt: start empty */
  }
}

function saveState() {
  try {
    const batches = state.batches.map((b) => ({
      ...b,
      cards: b.cards.map(({ busy, status, statusTone, ...card }) => card),
    }));
    localStorage.setItem(STORE_KEY, JSON.stringify({ batches, form: readForm() }));
  } catch {
    /* ignore quota / privacy mode */
  }
}

function allCards() {
  return state.batches.flatMap((b) => b.cards);
}

function findCard(id) {
  for (const batch of state.batches) {
    const card = batch.cards.find((c) => c.id === id);
    if (card) return card;
  }
  return null;
}

function matchesFilter(card, filter = state.filter) {
  switch (filter) {
    case "unreviewed": return !card.rating && !card.difficultyFeedback;
    case "good": return card.rating === "good";
    case "bad": return card.rating === "bad";
    default: return true;
  }
}

// ---------------------------------------------------------------------------------------
// api

async function api(path, { method = "GET", body, signal } = {}) {
  let response;
  try {
    response = await fetch(path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal,
    });
  } catch (error) {
    if (error.name === "AbortError") throw error;
    throw makeError("network");
  }
  let data = null;
  try {
    data = await response.json();
  } catch {
    data = null;
  }
  if (!response.ok) {
    const code = data?.error?.code || (response.status >= 500 ? "internal_error" : "invalid_request");
    throw makeError(code, data?.error);
  }
  return data;
}

function makeError(code, payload) {
  const [title, text] = ERROR_MESSAGES[code] || ["İstek başarısız", payload?.message || "Beklenmeyen bir hata oluştu."];
  const details = (payload?.details || []).map((d) => `${d.field}: ${d.message}`);
  const message = code === "invalid_request" && payload?.message && payload.message !== "Invalid request."
    ? payload.message
    : text;
  return new ApiError(code, title, message, details);
}

// ---------------------------------------------------------------------------------------
// dom helpers

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value === true ? "" : String(value));
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

function icon(name, extra = "") {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", `icon ${extra}`.trim());
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

function announce(message) {
  const node = $("#announcer");
  node.textContent = "";
  requestAnimationFrame(() => { node.textContent = message; });
}

const timeFormat = new Intl.DateTimeFormat("tr-TR", { hour: "2-digit", minute: "2-digit" });

function seconds(ms) {
  return Math.max(0, Math.round(ms / 1000));
}

// ---------------------------------------------------------------------------------------
// form

function readForm() {
  const form = $("#generate-form");
  return {
    language: form.language.value,
    category: form.category.value.trim(),
    difficulty: form.difficulty.value || null, // null: no difficulty is sent at all
    count: Number.parseInt(form.count.value, 10),
  };
}

function applyForm(values) {
  const form = $("#generate-form");
  if (values.language) form.language.value = values.language;
  if (typeof values.category === "string") form.category.value = values.category;
  if (values.difficulty !== undefined) {
    const radio = form.querySelector(`input[name="difficulty"][value="${CSS.escape(values.difficulty || "")}"]`);
    if (radio) radio.checked = true;
  }
  if (values.count) form.count.value = values.count;
}

function validateForm(values) {
  const form = $("#generate-form");
  for (const control of [form.category, form.count]) control.removeAttribute("aria-invalid");
  if (!Number.isInteger(values.count) || values.count < 1 || values.count > 30) {
    form.count.setAttribute("aria-invalid", "true");
    return { field: form.count, message: "Kart sayısı 1 ile 30 arasında olmalı." };
  }
  if (values.category && !/^[\p{L}\p{N}_ '’.,&-]{1,50}$/u.test(values.category)) {
    form.category.setAttribute("aria-invalid", "true");
    return { field: form.category, message: "Kategori yalnızca harf, rakam, boşluk ve - _ ' , . & içerebilir." };
  }
  return null;
}

function setFormBusy(busy) {
  const form = $("#generate-form");
  for (const control of form.querySelectorAll("input, select")) control.disabled = busy;
  const button = $("#generate-button");
  button.disabled = busy;
  button.setAttribute("aria-busy", String(busy));
  $(".btn__text", button).textContent = busy ? "Üretiliyor…" : "Kart üret";
}

// ---------------------------------------------------------------------------------------
// generation

async function generate(request) {
  if (state.running) return;
  hideError();
  state.lastRequest = request;
  const controller = new AbortController();
  const startedAt = performance.now();
  state.running = { startedAt, request, controller, interval: null };
  setFormBusy(true);
  showProgress(request);
  state.running.interval = setInterval(tickProgress, 250);
  tickProgress();

  const body = {
    language: request.language,
    count: request.count,
    category: request.category || null,
  };
  if (request.difficulty) body.difficulty = request.difficulty;

  try {
    const data = await api(API.generate, { method: "POST", body, signal: controller.signal });
    const elapsedMs = performance.now() - startedAt;
    const batch = {
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
      request,
      elapsedMs,
      cards: data.cards.map((card) => ({
        ...card,
        rating: null,
        difficultyFeedback: null,
        previous: [],
      })),
    };
    state.batches.unshift(batch);
    if (state.filter !== "all" && state.filter !== "unreviewed") setFilter("all", { silent: true });
    saveState();
    render({ enteringBatch: batch.id });
    const partial = batch.cards.length < request.count ? `, ${request.count} istenmişti` : "";
    announce(`${batch.cards.length} kart geldi${partial}.`);
    focusCard(batch.cards[0]?.id, { scroll: false });
  } catch (error) {
    if (error.name === "AbortError") {
      announce("Bekleme bırakıldı.");
    } else {
      showError(error, { retry: true });
    }
  } finally {
    clearInterval(state.running?.interval);
    state.running = null;
    hideProgress();
    setFormBusy(false);
  }
}

function showProgress(request) {
  const box = $("#progress");
  $("#progress-title").textContent = `${request.count} kart üretiliyor`;
  $("#progress-meta").textContent =
    [LANGUAGE_NAMES[request.language] || request.language, request.category || "genel", DIFFICULTY_NAMES[request.difficulty]]
      .filter(Boolean)
      .join(" · ");
  $("#progress-bar").style.transform = "scaleX(0)";
  $(".progress", box).dataset.overdue = "false";
  box.hidden = false;
  renderSkeleton(Math.min(request.count, 3));
}

function tickProgress() {
  if (!state.running) return;
  const elapsed = (performance.now() - state.running.startedAt) / 1000;
  const expected = expectedSeconds(state.running.request.count);
  $("#progress-timer").textContent = `${Math.floor(elapsed)} sn`;
  const ratio = Math.min(elapsed / expected, 0.96);
  $("#progress-bar").style.transform = `scaleX(${ratio.toFixed(3)})`;
  const note = $("#progress-note");
  const overdue = elapsed > expected * 2;
  $("#progress .progress").dataset.overdue = String(overdue);
  if (overdue) {
    note.dataset.tone = "attention";
    note.textContent = "Beklenenden uzun sürüyor. Model hâlâ çalışıyor olabilir; sunucu zaman aşımında kontrollü bir hata döner.";
  } else {
    note.dataset.tone = "";
    note.textContent = `Tahmini süre ~${expected} sn. Model kartları doğrular, gerekirse düzeltir.`;
  }
}

function hideProgress() {
  $("#progress").hidden = true;
  $("#skeleton-batch")?.remove();
  updateEmptyStates();
}

function renderSkeleton(rows) {
  $("#skeleton-batch")?.remove();
  const lines = Array.from({ length: rows }, () =>
    el("div", { class: "skeleton" },
      el("div", { class: "skeleton__line skeleton__line--title" }),
      el("div", { class: "skeleton__words" }, Array.from({ length: 5 }, () => el("div", { class: "skeleton__line" }))),
    ),
  );
  const skeleton = el("section", { class: "batch", id: "skeleton-batch", "aria-hidden": "true" },
    el("div", { class: "batch__header" }, el("span", { class: "batch__title", text: "Yeni kartlar bekleniyor" })),
    lines,
  );
  $("#batches").prepend(skeleton);
  updateEmptyStates();
}

// ---------------------------------------------------------------------------------------
// card actions

async function sendFeedback(card, kind, value) {
  const current = kind === "rating" ? card.rating : card.difficultyFeedback;
  if (current === value || card.busy) return;
  card.busy = kind;
  setRowStatus(card, "Kaydediliyor…", "");
  updateRow(card);
  const body = kind === "rating" ? { rating: value } : { difficulty_feedback: value };
  try {
    await api(API.feedback(card.id), { method: "POST", body });
    if (kind === "rating") card.rating = value;
    else card.difficultyFeedback = value;
    const label = kind === "rating" ? RATING_NAMES[value] : `zorluk ${DIFFICULTY_FEEDBACK[value].toLocaleLowerCase("tr")}`;
    setRowStatus(card, `Kaydedildi: ${label}`, "success");
    saveState();
  } catch (error) {
    setRowStatus(card, `${error.title}. ${error.text}`, "danger");
  } finally {
    card.busy = null;
    updateRow(card);
    updateCounts();
  }
}

async function regenerate(card) {
  if (card.busy) return;
  card.busy = "regenerate";
  setRowStatus(card, "Aynı hedef için yeni yasaklı kelimeler üretiliyor…", "");
  updateRow(card);
  const startedAt = performance.now();
  try {
    const fresh = await api(API.regenerate(card.id), { method: "POST" });
    card.previous = [
      { id: card.id, forbidden_words: card.forbidden_words, rating: card.rating },
      ...(card.previous || []),
    ].slice(0, 5);
    const oldId = card.id;
    Object.assign(card, {
      id: fresh.id,
      target: fresh.target,
      forbidden_words: fresh.forbidden_words,
      language: fresh.language,
      category: fresh.category,
      difficulty: fresh.difficulty,
      rating: null,
      difficultyFeedback: null,
    });
    if (state.selectedId === oldId) state.selectedId = card.id;
    setRowStatus(card, `Yenilendi (${seconds(performance.now() - startedAt)} sn). Eski kart artık kullanılmıyor.`, "success");
    saveState();
  } catch (error) {
    setRowStatus(card, `${error.title}. ${error.text}`, "danger");
  } finally {
    card.busy = null;
    updateRow(card, { refocus: true });
    updateCounts();
  }
}

async function copyCard(card) {
  const payload = {
    id: card.id,
    target: card.target,
    forbidden_words: card.forbidden_words,
    language: card.language,
    category: card.category,
    difficulty: card.difficulty,
  };
  try {
    await navigator.clipboard.writeText(JSON.stringify(payload, null, 2));
    setRowStatus(card, "JSON panoya kopyalandı", "success");
  } catch {
    setRowStatus(card, "Kopyalanamadı: tarayıcı panoya erişime izin vermedi", "danger");
  }
  updateRow(card);
}

function setRowStatus(card, text, tone) {
  card.status = text;
  card.statusTone = tone;
}

// ---------------------------------------------------------------------------------------
// rendering

function render({ enteringBatch = null } = {}) {
  const container = $("#batches");
  const skeleton = $("#skeleton-batch");
  container.replaceChildren(...(skeleton ? [skeleton] : []), ...state.batches.map((b) => renderBatch(b, b.id === enteringBatch)));
  updateCounts();
  applyFilter();
}

function renderBatch(batch, entering) {
  const request = batch.request;
  const returned = batch.cards.length;
  const createdAt = new Date(batch.createdAt);
  const facts = [
    el("span", {}, `${LANGUAGE_NAMES[request.language] || request.language}`),
    el("span", {}, request.category || "genel"),
  ];
  if (request.difficulty) facts.push(el("span", {}, DIFFICULTY_NAMES[request.difficulty] || request.difficulty));
  facts.push(el("span", {}, icon("clock", "icon--sm"), `${timeFormat.format(createdAt)} · ${seconds(batch.elapsedMs)} sn`));

  const header = el("div", { class: "batch__header" },
    el("h2", { class: "batch__title" }, `${returned} kart`),
    el("div", { class: "batch__facts" }, facts),
  );
  if (returned < request.count) {
    header.append(el("span", { class: "label label--attention", title: "Model istenen sayıda geçerli kart üretemedi." },
      icon("alert"), `${request.count} istendi, ${returned} geçerli`));
  }

  const list = el("ol", { class: "batch__list" }, batch.cards.map((card) => renderCard(card, entering)));
  return el("section", { class: "batch", "data-batch-id": batch.id, "aria-label": `${returned} kartlık üretim, ${timeFormat.format(createdAt)}` }, header, list);
}

function renderCard(card, entering = false) {
  const node = $("#card-template").content.firstElementChild.cloneNode(true);
  node.dataset.cardId = card.id;
  if (entering) node.dataset.entering = "true";
  if (state.selectedId === card.id) node.dataset.selected = "true";
  node.setAttribute("aria-busy", String(Boolean(card.busy)));
  const targetId = `target-${card.id}`;
  node.setAttribute("aria-labelledby", targetId);

  const lexeme = $(".card-row__lexeme", node);
  lexeme.lang = card.language;
  lexeme.dir = RTL_LANGUAGES.has(card.language) ? "rtl" : "ltr";
  const target = $(".card-row__target", node);
  target.id = targetId;
  target.textContent = card.target;
  $(".words", node).replaceChildren(
    ...card.forbidden_words.map((word, index) =>
      el("li", {}, el("span", { class: "words__n", "aria-hidden": "true" }, index + 1), word)),
  );

  const meta = $(".card-row__meta", node);
  meta.append(el("span", { class: "label" }, card.language));
  if (card.difficulty) meta.append(el("span", { class: "label" }, DIFFICULTY_NAMES[card.difficulty] || card.difficulty));
  if (card.category) meta.append(el("span", { class: "label" }, card.category));
  if (card.rating) {
    meta.append(el("span", { class: `label label--${card.rating === "good" ? "success" : "danger"}` },
      icon(card.rating === "good" ? "check" : "x"), RATING_NAMES[card.rating]));
  }
  if (card.difficultyFeedback) {
    meta.append(el("span", { class: "label label--accent" }, `Zorluk: ${DIFFICULTY_FEEDBACK[card.difficultyFeedback]}`));
  }
  if (card.previous?.length) {
    const last = card.previous[0];
    meta.append(el("span", { class: "card-row__previous" },
      el("span", { class: "label label--done" }, icon("history"), card.previous.length > 1 ? `${card.previous.length} kez yenilendi` : "Yenilendi"),
      el("span", {}, "Önceki: ", el("s", { lang: card.language }, last.forbidden_words.join(", "))),
    ));
  }

  for (const button of $$("[data-action]", node)) {
    const action = button.dataset.action;
    if (action === "good" || action === "bad") button.setAttribute("aria-pressed", String(card.rating === action));
    if (action in DIFFICULTY_FEEDBACK) button.setAttribute("aria-pressed", String(card.difficultyFeedback === action));
    button.disabled = Boolean(card.busy);
  }

  // "Too easy / right / too hard" only makes sense against a requested difficulty.
  $(".difficulty-feedback", node).hidden = !card.difficulty;

  const status = $(".card-row__status", node);
  if (card.status) {
    status.dataset.tone = card.statusTone || "";
    if (card.statusTone === "success") status.append(icon("check", "icon--sm"));
    if (card.statusTone === "danger") status.append(icon("alert", "icon--sm"));
    status.append(card.status);
  }
  if (!matchesFilter(card)) node.hidden = true;
  return node;
}

function rowFor(id) {
  return document.querySelector(`.card-row[data-card-id="${CSS.escape(id)}"]`);
}

function updateRow(card, { refocus = false } = {}) {
  // Rows are keyed by the card id at render time; regenerate changes the id, so look up
  // either the current id or the row that still carries the previous one.
  const previousId = card.previous?.[0]?.id;
  const row = rowFor(card.id) || (previousId && rowFor(previousId));
  if (!row) return;
  const active = document.activeElement;
  const hadFocus = row.contains(active);
  const action = hadFocus ? active.dataset?.action : null;
  const fresh = renderCard(card);
  row.replaceWith(fresh);
  if (hadFocus || refocus) {
    const target = action ? $(`[data-action="${action}"]`, fresh) : fresh;
    (target && !target.disabled ? target : fresh).focus({ preventScroll: true });
  }
}

function updateCounts() {
  const cards = allCards();
  const counts = {
    all: cards.length,
    unreviewed: cards.filter((c) => matchesFilter(c, "unreviewed")).length,
    good: cards.filter((c) => c.rating === "good").length,
    bad: cards.filter((c) => c.rating === "bad").length,
  };
  for (const [key, value] of Object.entries(counts)) {
    const counter = $(`[data-count="${key}"]`);
    if (counter) counter.textContent = value;
  }
  $("#total-counter").textContent = counts.all;
  $('[data-stat="total"]').textContent = counts.all;
  $('[data-stat="reviewed"]').textContent = counts.all - counts.unreviewed;
  $('[data-stat="good"]').textContent = counts.good;
  $('[data-stat="bad"]').textContent = counts.bad;
}

function applyFilter() {
  for (const batch of $$(".batch[data-batch-id]")) {
    let visible = 0;
    for (const row of $$(".card-row", batch)) {
      const card = findCard(row.dataset.cardId);
      const show = card ? matchesFilter(card) : true;
      row.hidden = !show;
      if (show) visible += 1;
    }
    batch.hidden = visible === 0;
  }
  updateEmptyStates();
}

function updateEmptyStates() {
  const hasCards = allCards().length > 0;
  const generating = Boolean($("#skeleton-batch"));
  $("#empty").hidden = hasCards || generating;
  const anyVisible = $$(".card-row").some((row) => !row.hidden);
  $("#filter-empty").hidden = !hasCards || anyVisible || generating;
}

function setFilter(filter, { silent = false } = {}) {
  state.filter = filter;
  for (const button of $$("#filter [data-filter]")) {
    button.setAttribute("aria-pressed", String(button.dataset.filter === filter));
  }
  applyFilter();
  if (!silent) announce(`Filtre: ${$(`#filter [data-filter="${filter}"]`).firstChild.textContent.trim()}`);
}

// ---------------------------------------------------------------------------------------
// selection & keyboard

function visibleRows() {
  return $$(".card-row").filter((row) => !row.hidden);
}

function focusCard(id, { scroll = true } = {}) {
  if (!id) return;
  state.selectedId = id;
  for (const row of $$(".card-row[data-selected]")) row.removeAttribute("data-selected");
  const row = rowFor(id);
  if (!row) return;
  row.dataset.selected = "true";
  row.focus({ preventScroll: !scroll });
  if (scroll) row.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function moveSelection(step) {
  const rows = visibleRows();
  if (!rows.length) return;
  const index = rows.findIndex((row) => row.dataset.cardId === state.selectedId);
  const next = index === -1 ? (step > 0 ? 0 : rows.length - 1) : Math.min(Math.max(index + step, 0), rows.length - 1);
  focusCard(rows[next].dataset.cardId);
}

function selectedCard() {
  const focusedRow = document.activeElement?.closest?.(".card-row");
  const id = focusedRow?.dataset.cardId || state.selectedId;
  return id ? findCard(id) : null;
}

function isTyping(target) {
  return target instanceof HTMLElement &&
    (target.isContentEditable || ["INPUT", "SELECT", "TEXTAREA"].includes(target.tagName));
}

function onKeydown(event) {
  const form = $("#generate-form");
  if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && form.contains(event.target)) {
    event.preventDefault();
    form.requestSubmit();
    return;
  }
  if (event.metaKey || event.ctrlKey || event.altKey || isTyping(event.target)) return;
  if ($("#shortcuts-dialog").open) return;

  const card = selectedCard();
  const handlers = {
    j: () => moveSelection(1),
    k: () => moveSelection(-1),
    n: () => form.language.focus(),
    "?": () => openShortcuts(),
    g: () => card && sendFeedback(card, "rating", "good"),
    b: () => card && sendFeedback(card, "rating", "bad"),
    1: () => card?.difficulty && sendFeedback(card, "difficulty", "too_easy"),
    2: () => card?.difficulty && sendFeedback(card, "difficulty", "correct"),
    3: () => card?.difficulty && sendFeedback(card, "difficulty", "too_hard"),
    r: () => card && regenerate(card),
    c: () => card && copyCard(card),
  };
  const handler = handlers[event.key];
  if (handler) {
    event.preventDefault();
    handler();
  }
}

function openShortcuts() {
  const dialog = $("#shortcuts-dialog");
  if (!dialog.open) dialog.showModal();
}

// ---------------------------------------------------------------------------------------
// errors

function showError(error, { retry = false } = {}) {
  $("#error-title").textContent = error.title || "İstek başarısız";
  const text = $("#error-text");
  text.textContent = error.text || "";
  $(".flash__details", $("#error-flash"))?.remove();
  if (error.details?.length) {
    text.after(el("ul", { class: "flash__details" }, error.details.map((d) => el("li", {}, d))));
  }
  $("#retry-button").hidden = !retry || !state.lastRequest;
  $("#error-flash").hidden = false;
}

function hideError() {
  $("#error-flash").hidden = true;
}

// ---------------------------------------------------------------------------------------
// health

async function refreshHealth() {
  const dot = $(".health__dot");
  const text = $("#health-text");
  const list = $("#health-list");
  const labels = { database: "Veritabanı", generation: "Kart üretimi (LLM)", embedding: "Embedding (RAG)" };
  const valueNames = { ok: "Çalışıyor", unavailable: "Erişilemiyor", disabled: "Kapalı" };
  const tones = { ok: "success", unavailable: "danger", disabled: "muted" };
  try {
    const data = await api(API.health);
    dot.dataset.state = data.status;
    text.textContent = { ok: "Sistem hazır", degraded: "Kısmen çalışıyor", unavailable: "Sistem erişilemiyor" }[data.status] || data.status;
    list.replaceChildren(...Object.entries(data.components).flatMap(([key, value]) => [
      el("dt", {}, labels[key] || key),
      el("dd", {}, el("span", { class: `label label--${tones[value] || "muted"}` }, valueNames[value] || value)),
    ]));
  } catch {
    dot.dataset.state = "unavailable";
    text.textContent = "Backend'e ulaşılamıyor";
    list.replaceChildren(el("dt", {}, "API"), el("dd", {}, el("span", { class: "label label--danger" }, "Yanıt yok")));
  }
  $("#health-checked").textContent = `Son kontrol ${timeFormat.format(new Date())} · her 30 sn yenilenir`;
}

// ---------------------------------------------------------------------------------------
// wiring

function onSubmit(event) {
  event.preventDefault();
  const values = readForm();
  const problem = validateForm(values);
  const errorNode = $("#form-error");
  if (problem) {
    errorNode.textContent = problem.message;
    errorNode.hidden = false;
    problem.field.focus();
    return;
  }
  errorNode.hidden = true;
  saveState();
  generate(values);
}

function onListClick(event) {
  const row = event.target.closest(".card-row");
  if (!row) return;
  const card = findCard(row.dataset.cardId);
  if (!card) return;
  const button = event.target.closest("[data-action]");
  if (!button) {
    if (!event.target.closest("a, button, input")) focusCard(card.id, { scroll: false });
    return;
  }
  state.selectedId = card.id;
  const action = button.dataset.action;
  if (action === "good" || action === "bad") sendFeedback(card, "rating", action);
  else if (action in DIFFICULTY_FEEDBACK) sendFeedback(card, "difficulty", action);
  else if (action === "regenerate") regenerate(card);
  else if (action === "copy") copyCard(card);
}

let clearTimer = null;
function onClear() {
  const button = $("#clear-button");
  if (button.dataset.confirm !== "true") {
    button.dataset.confirm = "true";
    $(".btn__text", button).textContent = "Emin misin? Tekrar tıkla";
    clearTimeout(clearTimer);
    clearTimer = setTimeout(resetClearButton, 4000);
    return;
  }
  resetClearButton();
  state.batches = [];
  state.selectedId = null;
  saveState();
  render();
  announce("Liste temizlendi.");
}

function resetClearButton() {
  const button = $("#clear-button");
  button.dataset.confirm = "false";
  $(".btn__text", button).textContent = "Listeyi temizle";
}

function init() {
  loadState();
  render();
  $("#generate-form").addEventListener("submit", onSubmit);
  $("#batches").addEventListener("click", onListClick);
  $("#batches").addEventListener("focusin", (event) => {
    const row = event.target.closest(".card-row");
    if (row && row.dataset.cardId !== state.selectedId) {
      state.selectedId = row.dataset.cardId;
      for (const other of $$(".card-row[data-selected]")) other.removeAttribute("data-selected");
      row.dataset.selected = "true";
    }
  });
  $("#filter").addEventListener("click", (event) => {
    const button = event.target.closest("[data-filter]");
    if (button) setFilter(button.dataset.filter);
  });
  $("#reset-filter").addEventListener("click", () => setFilter("all"));
  $("#cancel-button").addEventListener("click", () => state.running?.controller.abort());
  $("#retry-button").addEventListener("click", () => state.lastRequest && generate(state.lastRequest));
  $("#dismiss-error").addEventListener("click", hideError);
  $("#clear-button").addEventListener("click", onClear);
  $("#shortcuts-button").addEventListener("click", openShortcuts);
  $("#generate-form").addEventListener("change", saveState);
  document.addEventListener("keydown", onKeydown);
  refreshHealth();
  setInterval(refreshHealth, HEALTH_INTERVAL_MS);
}

init();
