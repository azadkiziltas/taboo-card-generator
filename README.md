# Taboo Card Backend

Çok dilli kelime anlatma oyunu (Flutter) için **modelden ve sağlayıcıdan bağımsız** AI kart
üretim backend'i. İlk sağlayıcı local **Ollama** (varsayılan model `gemma4:12b`); cloud sağlayıcılar
adapter yazarak veya yalnızca konfigürasyonla eklenir. Kart üretim kodu ve mobil API değişmez.

Ekip için bir kullanım arayüzü de gelir: **<http://localhost:8000/ui/>**. Kart üretme, inceleme,
iyi/kötü ve zorluk geri bildirimi ile yeniden üretme buradan yapılır.

- FastAPI + Pydantic v2 + SQLAlchemy 2 (async) + Alembic
- PostgreSQL + pgvector (RAG)
- Ollama (generation ve embedding ayrı modeller)
- Structured JSON çıktı, deterministik kart validasyonu, repair/retry, provider fallback
- Model registry (DB), routing policy (YAML), circuit breaker
- Versiyonlu prompt dosyaları, AI request telemetry (token, latency, maliyet), benchmark

---

## İçindekiler

1. [Mimari](#mimari)
2. [Hızlı başlangıç (Docker)](#hızlı-başlangıç-docker)
3. [Local geliştirme (Docker'sız API)](#local-geliştirme)
4. [Ollama](#ollama)
5. [PostgreSQL / pgvector ve migration](#postgresql--pgvector-ve-migration)
6. [API](#api)
7. [RAG](#rag)
8. [Model / provider değiştirme ve ekleme](#model--provider-değiştirme-ve-ekleme)
9. [Validasyon, repair ve retry limitleri](#validasyon-repair-ve-retry-limitleri)
10. [Promptlar](#promptlar)
11. [Observability](#observability)
12. [Testler ve benchmark](#testler-ve-benchmark)
13. [Konfigürasyon referansı](#konfigürasyon-referansı)
14. [Güvenlik](#güvenlik)
15. [Bilinen sınırlar ve sonraki adımlar](#bilinen-sınırlar-ve-sonraki-adımlar)

---

## Mimari

```text
Flutter ──> FastAPI (app/api) ──> Use case'ler (app/application)
                                   │
            ┌──────────────────────┼─────────────────────────┐
            v                      v                         v
      RAG (retrieve)        PromptService (dosyalar)   CardValidator + DuplicateDetector
            │                                                │
      EmbeddingRouter ──> EmbeddingProvider                  │
            │                                                │
        pgvector                         ModelRouter <───────┘
                                              │  registry + policy + circuit breaker
                                  ┌───────────┼────────────┐
                                  v           v            v
                               Ollama     Cloud A (*)   Cloud B (*)      (*) konfigürasyonla
```

**Katman kuralları** (`tests/unit/test_architecture.py` bunları otomatik kontrol eder):

| Katman | Klasör | Bilir | Bilmez |
|---|---|---|---|
| Domain | `app/domain` | kendi entity / value object / arayüzleri (+ Pydantic şemaları) | FastAPI, SQLAlchemy, httpx, herhangi bir LLM SDK'sı |
| Application | `app/application` | domain arayüzleri üzerinden orkestrasyon | altyapı implementasyonları, provider SDK'ları |
| Infrastructure | `app/infrastructure` | Ollama/HTTP adapterları, PostgreSQL, pgvector, telemetry | FastAPI route'ları |
| API | `app/api` | route'lar, DTO'lar, hata dönüşümleri | provider detayları |

`app/bootstrap.py` tüm somut implementasyonları bağlayan tek yerdir (composition root).

```text
backend/
  app/
    main.py                 FastAPI app factory
    bootstrap.py            composition root (DI)
    cli.py                  seed/RAG/generate komutları
    api/v1/                 public API: cards.py, health.py, schemas.py
    api/admin/              admin/debug API (token korumalı)
    api/errors.py           iç hatalar -> kontrollü public hata cevapları
    domain/cards/           Card, validator, normalization, duplicates, safety, şema
    domain/ai/              GenerationProvider, EmbeddingProvider, capabilities, routing,
                            exceptions, cost, telemetry
    domain/rag/             RagDocument, VectorStore arayüzü
    application/cards/      engine (üret→doğrula→repair), generate_cards, generate_card,
                            regenerate_card, submit_feedback
    application/ai/         model_router.py, embedding_router.py
    application/rag/        retrieve_context.py, index_document.py
    infrastructure/ai/      providers (ollama, openai_compatible, fake), registry, routing, config
    infrastructure/db/      SQLAlchemy modelleri, repository'ler, Alembic migration'ları
    infrastructure/rag/     pgvector_store.py
    infrastructure/observability/  telemetry repository, metrics, logging
    prompts/card/*.md       versiyonlu promptlar
  config/ai.yaml            provider'lar, registry seed, routing policy'leri
  data/rag/*.jsonl          örnek RAG verisi (tr, en)
  tests/unit | integration | benchmark
```

---

## Hızlı başlangıç (Docker)

Gereksinimler: Docker (Compose v2), host makinede [Ollama](https://ollama.com).

```bash
cd backend
cp .env.example .env              # DEFAULT_GENERATION_MODEL vb. düzenle, ADMIN_API_TOKEN ver

# 1) Modeller (host makinede)
ollama pull gemma4:12b            # varsayılan generation modeli (veya .env'de seçtiğin başka bir model)
ollama pull qwen3-embedding:0.6b  # çok dilli embedding modeli

# 2) API + PostgreSQL/pgvector (migration'lar container açılışında otomatik çalışır)
docker compose up -d --build
curl -s localhost:8000/v1/health
# {"status":"ok","components":{"database":"ok","generation":"ok","embedding":"ok"}}

# 3) Örnek RAG verisini yükle (embedding'ler Ollama ile hesaplanır)
docker compose exec api python -m app.cli rag-ingest data/rag/seed_tr.jsonl data/rag/seed_en.jsonl

# 4) Kart üret
curl -s -X POST localhost:8000/v1/cards/generate \
  -H 'Content-Type: application/json' \
  -d '{"language":"tr","category":"technology","difficulty":"medium","count":5}'
```

Kullanım arayüzü: <http://localhost:8000/ui/> · Swagger UI: <http://localhost:8000/docs> ·
OpenAPI: <http://localhost:8000/openapi.json>

**Arayüz** (`app/ui/static/`, build adımı yok, API ile aynı origin'den sunulur): soldaki formdan dil,
kategori, isteğe bağlı zorluk (varsayılan "Fark etmez") ve kart sayısı girilir. Kategori alanı serbest metindir; bir konu da yazılabilir
(`technology`, `uzay yolculuğu`, `90'lar müziği`). Gelen kartlar partiler hâlinde listelenir, her kart
iyi/kötü ve zorluk (kolay/uygun/zor) olarak işaretlenebilir ya da yeniden üretilebilir. Klavye:
`j`/`k` gezin, `g`/`b` iyi/kötü, `1`/`2`/`3` zorluk, `r` yeniden üret, `?` tüm kısayollar.
Liste tarayıcıda (localStorage) tutulur; kartlar ve geri bildirimler zaten veritabanına yazılır.

> API container'ı host'taki Ollama'ya `http://host.docker.internal:11434` üzerinden erişir
> (`OLLAMA_BASE_URL_DOCKER`). Ollama'yı container olarak çalıştırmak istersen:
> `docker compose --profile ollama up -d` ve `.env` içinde
> `OLLAMA_BASE_URL_DOCKER=http://ollama:11434`.

Portlar `.env` ile değiştirilebilir: `API_HOST_PORT` (8000), `POSTGRES_HOST_PORT` (5432).

---

## Local geliştirme

API'yi host'ta çalıştırmak (hot reload) için yalnızca veritabanını Docker'da başlat:

```bash
cd backend
uv sync                                  # Python 3.12 + bağımlılıklar (.venv)
cp .env.example .env
docker compose up -d postgres
uv run alembic upgrade head
uv run uvicorn app.main:create_app --factory --reload --port 8000
```

`uv` yoksa bağımlılıkları pip ile kur, sonra komutları `uv run` olmadan çalıştır:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install fastapi "uvicorn[standard]" pydantic-settings "sqlalchemy[asyncio]" asyncpg \
  alembic pgvector httpx pyyaml pytest pytest-asyncio
```

CLI komutları:

```bash
uv run python -m app.cli routing                         # router hangi modelleri, hangi sırayla dener
uv run python -m app.cli seed-models [--overwrite]       # config/ai.yaml -> ai_models tablosu
uv run python -m app.cli rag-ingest data/rag/seed_tr.jsonl
uv run python -m app.cli rag-search --language tr "mutfak aleti"
uv run python -m app.cli rag-reindex                     # aktif embedding space'te eksik vektörler
uv run python -m app.cli generate --language tr --category food --count 3
uv run python -m app.cli generate-for-target --language tr --target uçak --no-persist
```

---

## Ollama

- Generation modeli: `DEFAULT_GENERATION_MODEL` (varsayılan `gemma4:12b`). Model adı kodda geçmez;
  `.env` → `config/ai.yaml` → `ai_models` registry tablosu zinciriyle gelir.
- Embedding modeli: `DEFAULT_EMBEDDING_MODEL` (örn. `qwen3-embedding:0.6b`, 1024 boyut, çok dilli).
- Structured output: Ollama `format` alanına JSON Schema gönderilir (native schema). Model bunu
  desteklemiyorsa registry'de `supports_structured_output=false` yapılır; router otomatik olarak
  JSON mode'a ya da strict prompt + tolerant JSON parser'a düşer.
- Düşünen (reasoning) modeller için `OLLAMA_THINK=false` (varsayılan) düşünme çıktısını kapatır.

**Model seçimi.** Benchmark runner ile gerçek pipeline üzerinde ölçüldü (Apple M4 Pro, 24 GB;
5 Türkçe hedef kelime + TR/AR/EN toplu üretim, aynı promptlar ve RAG):

| Model | Tek kart (ort.) | 5 kart toplu | İlk geçişte validasyon | Not |
|---|---|---|---|---|
| **`gemma4:12b`** (varsayılan) | 6.2 sn | ~19 sn | %100 | Türkçe kelimelerde hata yok |
| `qwen3.5:9b` | 4.9 sn | ~16 sn | %92 | yazım hataları ("odunç", uydurma "gözükçe") |
| `qwen3.5:27b` | ~28 sn | ~70 sn | %100 | GPU belleğine sığmıyor; bellek baskısında 180 sn timeout |

Canlı API üzerinden `gemma4:12b` ile 5 kart ~26 sn, 10 kart ~31 sn sürer. Başka bir modeli denemek
için aynı ölçümü çalıştır ([Testler ve benchmark](#testler-ve-benchmark)):

```bash
uv run python -m tests.benchmark.runner --dataset tests/benchmark/datasets/tr_basic.json --limit 5 \
  --model ollama:gemma4:12b --model ollama:<aday-model> \
  --rag-seed data/rag/seed_tr.jsonl --batch tr:technology:5 --batch ar:animals:3
```

**Bellek notu.** Generation modeli ve embedding modeli birlikte GPU belleğine sığmalıdır
(`ollama ps` → `100% GPU`). Model sığmazsa Ollama bir kısmını CPU'ya taşır, istekler dakikalar sürer.
GPU bellek yetersizliğinde (Metal OOM) de Ollama HTTP 200 ile boş cevap döner. Backend bu durumu
`ProviderUnavailable` sayar ve fallback ile circuit breaker devreye girer. Büyük bir model kullanacaksan:

```env
OLLAMA_NUM_CTX=4096            # 10 kartlık batch için yeterli
OLLAMA_EMBED_KEEP_ALIVE=0      # embedding modelini her çağrıdan sonra bellekten çıkar
```

Hata durumuna düşmüş bir runner'ı sıfırlamak için: `ollama stop <model>`. Model değiştirdikten sonra
mevcut registry satırlarını güncellemek için `python -m app.cli seed-models --overwrite` çalıştır;
kullanılmayan modeli `PATCH /admin/v1/models/{id}` ile `{"enabled": false}` yapabilirsin.

---

## PostgreSQL / pgvector ve migration

`pgvector/pgvector:pg17` imajı kullanılır; ilk migration `CREATE EXTENSION vector` çalıştırır.

```bash
uv run alembic upgrade head                        # en son şemaya geç
uv run alembic current                             # mevcut revizyon
uv run alembic revision --autogenerate -m "..."    # model değişikliğinden yeni migration
uv run alembic downgrade -1
```

Tablolar:

| Tablo | İçerik |
|---|---|
| `cards` | kart (target, normalize `target_key`, dil, kategori, zorluk, kaynak, durum, provenance) |
| `card_forbidden_words` | 5 yasaklı kelime (`position` 1-5, normalize `word_key`) |
| `card_feedback` | oyuncu geri bildirimi (rating, difficulty_feedback), PII yok |
| `rag_documents` | RAG dokümanları (dil, kelime, içerik, kategori, ilişkili terimler, kaynak, lisans, metadata) |
| `rag_embeddings` | doküman vektörleri, **embedding space** başına (provider + model + version + dimension) |
| `ai_models` | model registry (capabilities, fiyatlar, diller, quality_score, latency, health) |
| `ai_requests` | her AI çağrısı: task, provider, model, prompt versiyonu, token, `api_estimated_cost`, latency, hata tipi, retry, fallback, parse hatası, validation sonuçları |
| `generation_runs` | her üretim isteği: istenen/dönen kart, draft sayısı, kabul oranı, repair, RAG doküman sayısı |

**Embedding space'ler.** Vektörler hangi model/versiyonla üretildiğiyle birlikte saklanır ve
arama her zaman tek bir space içinde yapılır; farklı modellerin vektörleri asla karışmaz.
Embedding modelini değiştirmek için:

```bash
# .env: DEFAULT_EMBEDDING_MODEL=<yeni-model>  EMBEDDING_VERSION=v2
uv run python -m app.cli rag-reindex     # yeni space için tüm dokümanları embed eder
```

Her space için `rag-ingest`/`rag-reindex` sonrasında kısmi HNSW index otomatik oluşturulur.

---

## API

### Public (Flutter)

Flutter hiçbir zaman provider veya model seçmez. İstekte bilinmeyen alanlar (`"model"` gibi)
**422** ile reddedilir. Cevaplarda provider/model bilgisi bulunmaz.

**`POST /v1/cards/generate`**

```bash
curl -s -X POST localhost:8000/v1/cards/generate -H 'Content-Type: application/json' -d '{
  "language": "tr",
  "category": "technology",
  "difficulty": "medium",
  "count": 10,
  "topic": null
}'
```

```json
{
  "cards": [
    {
      "id": "6320ba57-4070-44c4-954c-a78f5a564a25",
      "target": "Yapay Zeka",
      "forbidden_words": ["Algoritma", "Bilgisayar", "Öğrenme", "Robot", "Simülasyon"],
      "language": "tr",
      "category": "technology",
      "difficulty": "medium"
    }
  ]
}
```

- `language`: ISO-639-1 (`tr`, `en`, `de`, `fr`, `es`, `ar`; `SUPPORTED_LANGUAGES` ile ayarlanır)
- `difficulty`: isteğe bağlı, `easy | medium | hard`. Gönderilmezse modele zorlukla ilgili hiçbir şey
  iletilmez (prompt'ta da model şemasında da yok) ve kart `"difficulty": null` olarak döner/saklanır.
  Zorluk geri bildirimi (`too_easy/correct/too_hard`) yalnızca zorluklu kartlar için anlamlıdır.
- `count`: 1..`MAX_CARDS_PER_REQUEST`
- `category`: serbest metin, kategori ya da konu (en fazla 50 karakter; harf, rakam, boşluk ve `- _ ' , . &`).
  Dile duyarlı küçük harfe çevrilip saklanır (`İstanbul` → `istanbul`).
- `topic`: opsiyonel ek tema (en fazla 100 karakter). Arayüz bunu kullanmaz; kategori alanı konu da alır.
- Sistem tam `count` geçerli kart üretmeye çalışır. `ALLOW_PARTIAL_RESULTS=true` iken eksik
  sayıda kart dönebilir; hiç geçerli kart yoksa 502 döner.

**`POST /v1/cards/{id}/feedback`**

```bash
curl -s -X POST localhost:8000/v1/cards/<id>/feedback -H 'Content-Type: application/json' \
  -d '{"rating": "good", "difficulty_feedback": "too_easy"}'
# 201 {"id": "...", "card_id": "...", "status": "recorded"}
```

`rating`: `good | bad` · `difficulty_feedback`: `too_easy | correct | too_hard` (en az biri gerekli).

**`POST /v1/cards/{id}/regenerate`**: aynı hedef kelime için yeni yasaklı kelimeler üretir;
eski kart `replaced` durumuna geçer.

**`GET /v1/health`**: yalnızca kaba bileşen durumu döner (`ok | degraded | unavailable`), provider detayı içermez.

**Hata formatı** (tüm hatalar):

```json
{"error": {"code": "generation_unavailable", "message": "Card generation is temporarily unavailable. Please retry."}}
```

| HTTP | code | Ne zaman |
|---|---|---|
| 422 | `invalid_request` | doğrulama hatası, desteklenmeyen dil, bilinmeyen alan |
| 404 | `card_not_found` | |
| 429 | `rate_limited` | rate limiter (şimdilik no-op extension point) |
| 502 | `generation_failed` | model geçerli JSON/kart üretemedi (retry/repair sonrası) |
| 503 | `generation_unavailable` / `generation_busy` / `storage_unavailable` / `context_unavailable` | provider, DB veya RAG erişilemez |
| 504 | `generation_timeout` | |

### Admin (`/admin/v1`, `X-Admin-Token` header'ı gerekli)

`ADMIN_API_TOKEN` boşsa admin API kapalıdır (403).

```bash
T=$(grep ^ADMIN_API_TOKEN .env | cut -d= -f2)
curl -s localhost:8000/admin/v1/models -H "X-Admin-Token: $T"                     # registry
curl -s -X PATCH localhost:8000/admin/v1/models/<id> -H "X-Admin-Token: $T" \
  -H 'Content-Type: application/json' -d '{"enabled": false}'                    # model kapat/aç, fiyat, priority, quality_score
curl -s localhost:8000/admin/v1/routing/card_generation -H "X-Admin-Token: $T"    # denenecek model sırası
curl -s localhost:8000/admin/v1/health -H "X-Admin-Token: $T"                     # provider sağlığı, circuit breaker
curl -s "localhost:8000/admin/v1/telemetry?hours=24" -H "X-Admin-Token: $T"       # token, maliyet, latency, kabul oranı
curl -s localhost:8000/admin/v1/metrics -H "X-Admin-Token: $T"                    # process içi sayaçlar
curl -s -X POST localhost:8000/admin/v1/rag/documents -H "X-Admin-Token: $T" -H 'Content-Type: application/json' \
  -d '{"documents":[{"language":"tr","word":"kahve","content":"Kavrulmuş çekirdeklerden yapılan içecek","source":"manual"}]}'
curl -s -X POST localhost:8000/admin/v1/rag/search -H "X-Admin-Token: $T" -H 'Content-Type: application/json' \
  -d '{"language":"tr","query":"sıcak içecek"}'
curl -s -X POST localhost:8000/admin/v1/cards/generate-for-target -H "X-Admin-Token: $T" -H 'Content-Type: application/json' \
  -d '{"target":"uçak","language":"tr","difficulty":"easy","persist":false}'
```

---

## RAG

Akış: `topic/category/target → query → EmbeddingProvider → pgvector (aynı dil filtresi) → top-k → prompt'ta <reference_context>`.

- `RAG_ENABLED=false` ile tamamen kapanır.
- `RAG_TOP_K` ve `RAG_MIN_SIMILARITY` ayarlanabilir.
- `RAG_FAIL_OPEN=true` (varsayılan): embedding veya DB hatasında bağlamsız devam edilir. `false` olursa 503 döner.
- RAG içeriği prompt'a **talimat olarak değil**, işaretli ve güvenilmeyen bağlam olarak girer. System prompt
  bu bloktaki talimatların yok sayılmasını söyler; `<`/`>` karakterleri temizlenir.
- Arama her zaman istek dilinde yapılır (TR→TR, EN→EN). Çeviri tabanlı üretim yoktur.

JSONL doküman formatı (`data/rag/seed_tr.jsonl`):

```json
{"language":"tr","word":"uçak","content":"Kanatları sayesinde havada uçan taşıt.","category":"transport","related_terms":["pilot","havalimanı"],"source":"project-seed","source_license":"CC0-1.0","metadata":{}}
```

Aynı `(language, word, source)` tekrar yüklenirse doküman güncellenir. İçerik değiştiyse vektörü de yeniden hesaplanır.

---

## Model / provider değiştirme ve ekleme

**Başka bir Ollama modeli:** `.env` → `DEFAULT_GENERATION_MODEL=<model>`, `ollama pull <model>`,
yeniden başlat. Registry'ye otomatik eklenir (`MODEL_REGISTRY_SYNC=insert_missing`).

**Fallback modeli:** `.env` → `FALLBACK_GENERATION_MODEL=<ikinci-model>`. Router önce primary'yi
dener. Bağlantı, timeout, 5xx/429 veya parse hatasında sıradakine geçer
(`max_provider_attempts`). Sürekli hata veren model circuit breaker ile bir süre devreden çıkar.

**OpenAI-uyumlu bir cloud sağlayıcı** (kod değişikliği gerekmez), `config/ai.yaml` üzerinden:

```yaml
providers:
  cloud_a:
    type: openai_compatible
    base_url: https://api.cloud-a.example/v1
    api_key_env: CLOUD_A_API_KEY        # anahtarın kendisi değil, env değişkeninin ADI
models:
  - provider: cloud_a
    model_name: model-x
    model_type: generation
    is_local: false
    priority: 30
    capabilities: {structured_output: true, json_mode: true}
    context_window: 128000
    pricing: {input_cost_per_million: 0.10, output_cost_per_million: 0.40, currency: USD}
routing:
  card_generation:
    providers:
      - {provider: ollama,  model: "${DEFAULT_GENERATION_MODEL}", priority: 1}
      - {provider: cloud_a, model: model-x, priority: 2}
```

`.env` → `CLOUD_A_API_KEY=...`. Anahtar yoksa provider devre dışı kalır ve uygulama yine başlar
(`/admin/v1/health` → `disabled_providers`). Registry satırı zaten varsa
`python -m app.cli seed-models --overwrite` çalıştır ya da admin API ile güncelle.

**Yeni bir provider tipi** (farklı bir API), card kodu değişmeden:

1. `app/infrastructure/ai/providers/` altında `BaseGenerationProvider`'dan türet; yalnızca `_complete()` ve `health_check()` yaz. HTTP hatalarını `send_json` normalize eder.
2. `app/infrastructure/ai/providers/factory.py` → `GENERATION_FACTORIES` (ve gerekiyorsa `EMBEDDING_FACTORIES`) içine kaydet.
3. `config/ai.yaml` → `models` altında registry'ye ekle (capabilities + fiyat).
4. `config/ai.yaml` → `routing` altında policy'de aktif et.

Routing stratejileri (`strategy`): `priority` (varsayılan), `cost`, `quality`, `latency`, `local_first`.
Filtreler: enabled, adapter mevcut mu, circuit durumu, capability (`structured_output`), dil, local_only.

---

## Validasyon, repair ve retry limitleri

```text
LLM çıktısı → Pydantic şema → Unicode normalizasyon → kural validator'ları → güvenlik filtresi
            → duplicate kontrolü (batch + DB) → geçerli: kaydet | onarılabilir: repair prompt | değil: at
```

Kurallar (`app/domain/cards/validator.py`, her biri ayrı ve değiştirilebilir `CardRule`):
tam 5 yasaklı kelime, boş değer yok, tekrar eden yasaklı kelime yok (dile duyarlı: `IŞIK` = `ışık`),
target yasaklılar arasında yok, yasaklı kelime target'ı içermiyor (`kahve` → `kahveci`),
maksimum uzunluk, dilin alfabesi (Latin/Arapça script kontrolü), aynı batch'te tekrar eden target yok,
DB'de aynı target + benzer yasaklı set (Jaccard ≥ `DUPLICATE_FORBIDDEN_OVERLAP_THRESHOLD`) → reddedilir,
aynı target farklı set → `duplicate_candidate` olarak işaretlenir.

Normalizasyon dile göre genişletilebilir (`NormalizerRegistry`): Türkçe/Azerice I/İ kuralları,
Arapça hareke/tatweel/elif varyantları. Arapça metin olduğu gibi saklanır; RTL gösterimi Flutter'ın işidir.

Güvenlik filtresi: `config/blocklists/<dil>.txt` (satır başına bir terim) varsa devreye girer.
`ContentSafetyFilter` arayüzüyle harici bir moderasyon servisi takılabilir.

Limitler (hepsi `.env` ile değiştirilebilir, sonsuz döngü yok):

| Ayar | Varsayılan | Anlamı |
|---|---|---|
| `MAX_PARSE_RETRIES` | 1 | aynı modelde bozuk JSON sonrası "JSON'u düzelt" denemesi |
| `MAX_REPAIR_ATTEMPTS` | 2 | validasyondan kalan kartlar için repair prompt turu (target korunur) |
| `MAX_GENERATION_ROUNDS` | 4 | eksik kart için yeni üretim turu |
| `max_provider_attempts` (ai.yaml) / `DEFAULT_MAX_PROVIDER_ATTEMPTS` | 3 | fallback zinciri uzunluğu |
| `CARD_BATCH_SIZE` | 10 | tek LLM çağrısında istenen en fazla kart |

---

## Promptlar

`app/prompts/<grup>/<ad>_<versiyon>.md`, YAML front matter ile (`name`, `version`,
`language_strategy`, `created_at`). Yer tutucular `{{ değişken }}` biçimindedir. Eksik ya da fazla
değişken hata verir. Promptlarda provider/model adı geçmez (test ile kontrol edilir).

| Dosya | Kullanım |
|---|---|
| `card/system_v1.md` | kurallar + prompt injection koruması |
| `card/generate_v1.md` | kategori/zorluk için yeni kartlar |
| `card/generate_for_targets_v1.md` | verilen hedef kelime(ler) için kart (regenerate, benchmark) |
| `card/repair_v1.md` | validasyon hatalarıyla birlikte düzeltme |
| `common/json_schema_instruction_v1.md` | native schema desteklemeyen modeller için |
| `common/json_repair_v1.md` | parse hatası sonrası |

Yeni versiyon: `generate_v2.md` gibi dosyalar ekle (front matter `version: v2`),
`CARD_PROMPT_VERSION=v2`. Kullanılan prompt adı/versiyonu her `ai_requests` ve `generation_runs` satırına yazılır.

---

## Observability

- Her provider çağrısı (parse retry'lar dahil) `ai_requests`'e yazılır: task, provider, model,
  local/cloud, prompt versiyonu, token, `api_estimated_cost` (registry fiyatlarından; local = 0 API
  maliyeti, altyapı maliyeti değil), latency, hata tipi, retry, fallback, parse hatası, validation
  hatası ve kabul edilen kart sayısı.
- Her istek `generation_runs`'a yazılır (kart kabul oranı, repair, tur sayısı, RAG doküman sayısı).
- Kartlar `ai_request_id` / `generation_run_id` ile üretildikleri çağrıya bağlanır. Bu bağlantı
  ileride SFT/fine-tuning verisi (prompt versiyonu + model + feedback) için kullanılabilir.
- Loglar key=value formatındadır. Prompt, model cevabı ve API anahtarları loglanmaz.
- `GET /admin/v1/telemetry` ve `GET /admin/v1/metrics` özet verir.

---

## Testler ve benchmark

```bash
uv run pytest                              # unit + (PostgreSQL açıksa) integration
uv run pytest tests/unit                   # LLM ve DB gerektirmez (fake provider'lar)
docker compose up -d postgres && uv run pytest tests/integration
RUN_OLLAMA_TESTS=1 uv run pytest tests/integration/test_ollama_live.py   # gerçek Ollama (yavaş)
uv run ruff check app tests
```

- Unit testler: validator, normalizasyon, duplicate, structured output parsing, capability seçimi,
  routing, fallback, parse retry, circuit breaker, maliyet hesabı, prompt rendering,
  provider adapterları (mock HTTP), use case'ler, HTTP sözleşmesi, mimari sınırlar.
- Integration testler ayrı bir `<db>_test` veritabanı oluşturup Alembic ile migrate eder;
  development verisine dokunmaz. PostgreSQL yoksa atlanır.
- Gerçek cloud çağrısı yapan test yoktur.

**Benchmark** (`tests/benchmark/`): verilen hedef kelimeleri gerçek pipeline'dan geçirir, router'ı
tek bir modele sabitler, veritabanına yazmaz.

```bash
uv run python -m tests.benchmark.runner \
  --dataset tests/benchmark/datasets/tr_basic.json \
  --model ollama:gemma4:12b [--model cloud_a:model-x] \
  --rag-seed data/rag/seed_tr.jsonl [--limit 5] [--batch tr:technology:5]
# -> benchmark-reports/<zaman>/report.json, summary.csv, items.csv, batches.csv
```

Model başına ölçülenler: final başarı oranı, JSON/şema başarı oranı, ilk geçişte validasyon oranı,
duplicate oranı, retry oranı, ortalama/p95 latency, token, tahmini API maliyeti; `--batch` ile
arayüzün kullandığı toplu üretimin süresi ve dönen kart oranı. `items.csv`'deki
boş `human_rating` sütunu insan puanlaması içindir. `quality_score` otomatik uydurulmaz:
`QualityScorer` extension point'i vardır; değer admin API ile (`PATCH /admin/v1/models/{id}`) girilir.

---

## Konfigürasyon referansı

Tamamı `.env.example` içinde açıklamalı olarak bulunur. En önemlileri:

| Değişken | Açıklama |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://...` (compose'da container için otomatik ayarlanır) |
| `DEFAULT_GENERATION_PROVIDER` / `_MODEL` | birincil generation modeli |
| `FALLBACK_GENERATION_PROVIDER` / `_MODEL` | opsiyonel ikinci model |
| `DEFAULT_EMBEDDING_PROVIDER` / `_MODEL`, `EMBEDDING_VERSION` | aktif embedding space |
| `OLLAMA_BASE_URL`, `OLLAMA_BASE_URL_DOCKER`, `OLLAMA_TIMEOUT_SECONDS` | Ollama erişimi |
| `OLLAMA_THINK`, `OLLAMA_NUM_CTX`, `OLLAMA_KEEP_ALIVE`, `OLLAMA_EMBED_KEEP_ALIVE` | model seçenekleri (registry `provider_options`) |
| `SUPPORTED_LANGUAGES` | kabul edilen diller |
| `RAG_ENABLED`, `RAG_TOP_K`, `RAG_MIN_SIMILARITY`, `RAG_FAIL_OPEN` | RAG |
| `MODEL_REGISTRY_SYNC` | `insert_missing` (varsayılan) · `overwrite` · `off` |
| `ADMIN_API_TOKEN` | admin API anahtarı (boş = kapalı) |
| `CORS_ALLOW_ORIGINS` | Flutter web için (mobil için gerekmez) |

---

## Güvenlik

- Secret'lar yalnızca `.env` / secret manager'da tutulur. `.env` git'e girmez (`.gitignore`), `ai.yaml`
  yalnızca env değişkeninin **adını** içerir.
- İstek doğrulama: dil deseni, sayı sınırları, kategori karakter seti, `topic` uzunluğu, bilinmeyen alanlar reddedilir.
- Provider hata gövdeleri ve iç adresler istemciye dönmez (test ile kontrol edilir).
- SQL: yalnızca ORM ve parametreli sorgular (tek istisna olan index DDL'i literal escaping kullanır).
- Prompt injection: system talimatı ile RAG bağlamı / kullanıcı teması ayrı ve işaretli bloklardadır.
  RAG metni talimat olarak yorumlanmaz.
- Admin API ayrı prefix'te ve token korumalıdır. Public health provider detayı vermez.
- Rate limiting için `RateLimiter` arayüzü hazırdır (şu an no-op).
- Container non-root kullanıcıyla çalışır.

---

## Bilinen sınırlar ve sonraki adımlar

- **Senkron üretim:** `gemma4:12b` ile 5-10 kart 20-30 sn sürer, büyük modellerde dakikalar. Production için
  arka plan işi + önceden üretilmiş kart havuzu (örn. Redis/queue) önerilir.
- Circuit breaker ve metrikler süreç içidir. Birden fazla worker/instance için Redis tabanlı
  implementasyon aynı arayüzle takılabilir.
- Semantik duplicate kontrolü için yalnızca arayüz var (`SemanticDuplicateChecker`).
- Morfolojik analiz yok. `kitap`/`kitabı` gibi çekimler yakalanmaz, validator'a yeni bir kural olarak eklenebilir.
- Blocklist dosyaları boş gelir; içerik politikası ürün kararıdır.
- Örnek RAG verisi küçüktür (`project-seed`, CC0). Gerçek sözlük verisi eklerken `source_license` alanını doldur.
