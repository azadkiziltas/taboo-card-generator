# Model-Agnostic AI Card Generation Backend --- Implementation Specification

## 0. Amaç

Bu dokümanın amacı, Flutter ile geliştirilecek çok dilli bir kelime
anlatma oyunu için **modelden ve sağlayıcıdan bağımsız AI backend
mimarisi** oluşturmaktır.

Backend şu özelliklere sahip olmalıdır:

-   Local LLM çalıştırabilmeli (ilk geliştirme ortamı: Ollama + Qwen).
-   Cloud LLM sağlayıcıları sonradan adapter yazarak eklenebilmeli.
-   LLM sağlayıcısı/modeli değiştirilince kart üretim servisi ve mobil
    API değişmemeli.
-   Generation modeli ile embedding modeli birbirinden tamamen
    ayrılmalı.
-   RAG desteklemeli.
-   Çoklu dil desteklemeli.
-   Structured JSON çıktı üretmeli.
-   Kart çıktısını deterministic kurallarla doğrulamalı.
-   Retry ve provider fallback desteklemeli.
-   Model registry ve routing policy desteklemeli.
-   Promptlar versiyonlanmalı.
-   Model maliyeti, latency, hata ve kalite metrikleri kaydedilmeli.
-   İleride fine-tuning/SFT için kaliteli veri toplanabilmeli.
-   İlk sürüm ücretsiz/local geliştirmeye uygun olmalı.
-   Mimari production ortamına taşınabilir olmalı.

> Temel ilke: Domain katmanı `Ollama`, `Qwen`, herhangi bir cloud
> sağlayıcı veya belirli bir SDK'yı bilmemelidir.

------------------------------------------------------------------------

# 1. Önerilen Teknolojiler

İlk implementasyon:

-   Python 3.12+
-   FastAPI
-   Pydantic v2
-   SQLAlchemy 2.x
-   Alembic
-   PostgreSQL
-   pgvector
-   httpx
-   Ollama
-   pytest
-   Docker / Docker Compose
-   Redis: ilk MVP için zorunlu değil, daha sonra cache/queue/rate limit
    için eklenebilir.

Asenkron I/O tercih et.

------------------------------------------------------------------------

# 2. Yüksek Seviye Mimari

``` text
Flutter
   |
   v
FastAPI
   |
   v
Card Application Service
   |
   +----------------+----------------+
   |                |                |
   v                v                v
RAG Service    Prompt Service    Card Validator
   |                                 |
   v                                 |
Embedding Router                     |
   |                                 |
   v                                 |
pgvector                             |
                                     |
                    +----------------+
                    v
               Model Router
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Ollama    Cloud A   Cloud B
          |
        Qwen

Model Router
   |
   +--> Model Registry
   +--> Routing Policies
   +--> Provider Health
   +--> Capability Normalization

Observability
   |
   +--> requests
   +--> tokens
   +--> estimated cost
   +--> latency
   +--> retries
   +--> validation failures
   +--> selected provider/model
```

------------------------------------------------------------------------

# 3. Katman Kuralları

## Domain

Saf iş kurallarını içerir.

Domain katmanı şunları import ETMEMELİ:

-   ollama
-   OpenAI SDK
-   herhangi bir cloud LLM SDK
-   FastAPI
-   SQLAlchemy

Domain yalnızca kendi entity/value object/interface yapılarını
bilmelidir.

## Application

Use-case orchestration:

-   kart üret
-   kartı yeniden üret
-   RAG bağlamı getir
-   model seç
-   doğrula
-   kaydet

## Infrastructure

Dış sistem implementasyonları:

-   Ollama adapter
-   cloud provider adapterları
-   PostgreSQL
-   pgvector
-   HTTP client
-   telemetry

## API

FastAPI route, request/response DTO ve hata dönüşümleri.

------------------------------------------------------------------------

# 4. Proje Dizini

Aşağıdaki yapıya yakın bir proje oluştur:

``` text
app/
  main.py

  api/
    v1/
      cards.py
      health.py
      models.py
    dependencies.py
    errors.py

  domain/
    cards/
      entities.py
      schemas.py
      validator.py
      repositories.py
    ai/
      generation.py
      embeddings.py
      capabilities.py
      routing.py
      exceptions.py

  application/
    cards/
      generate_card.py
      generate_cards.py
      regenerate_card.py
    rag/
      retrieve_context.py
      index_document.py
    ai/
      model_router.py
      embedding_router.py

  infrastructure/
    ai/
      providers/
        ollama_generation.py
        ollama_embedding.py
        cloud_example.py
      registry/
        database_registry.py
      routing/
        policy_router.py

    db/
      models/
      repositories/
      session.py
      migrations/

    rag/
      pgvector_store.py

    observability/
      metrics.py
      ai_usage_repository.py

  prompts/
    card/
      system_v1.md
      generate_v1.md
      repair_v1.md

  config/
    settings.py

tests/
  unit/
  integration/
  benchmark/

docker-compose.yml
Dockerfile
.env.example
README.md
```

İsimlerde küçük farklılıklar yapılabilir ancak bağımlılık sınırlarını
koru.

------------------------------------------------------------------------

# 5. Domain Kart Modeli

Kart modeli provider/model bilgisinden bağımsız olmalıdır.

``` python
class Card:
    id: UUID | None
    target: str
    forbidden_words: list[str]
    language: str
    category: str | None
    difficulty: str
    source: str
```

Kurallar:

-   `target` boş olamaz.
-   `forbidden_words` tam 5 eleman içermeli.
-   Yasaklı kelimeler boş olamaz.
-   Aynı yasaklı kelime tekrar edemez.
-   Target kelimesinin birebir aynısı yasaklı kelimelerde bulunamaz.
-   Dil ISO-639-1 kodu şeklinde tutulmalı: `tr`, `en`, `de`, `fr`, `es`,
    `ar`.
-   `difficulty`: `easy | medium | hard`.
-   Unicode normalization uygulanmalı.
-   Türkçe büyük/küçük harf gibi dile özgü karşılaştırmalar için
    genişletilebilir normalization katmanı oluştur.

İlk sürümde morfolojik analiz zorunlu değil; fakat validator
genişletilebilir tasarlanmalı.

------------------------------------------------------------------------

# 6. Public API Sözleşmesi

Flutter hiçbir zaman provider/model seçmemelidir.

## POST `/v1/cards/generate`

Request:

``` json
{
  "language": "tr",
  "category": "technology",
  "difficulty": "medium",
  "count": 10,
  "topic": null
}
```

Response:

``` json
{
  "cards": [
    {
      "id": "uuid",
      "target": "Bilgisayar",
      "forbidden_words": [
        "ekran",
        "klavye",
        "internet",
        "işlemci",
        "yazılım"
      ],
      "language": "tr",
      "category": "technology",
      "difficulty": "medium"
    }
  ]
}
```

Response içerisinde provider/model bilgisini Flutter'ın oyun mantığı
için zorunlu hale getirme.

Gerekirse admin/debug telemetry ayrı endpointte bulunabilir.

## POST `/v1/cards/{id}/feedback`

İleride eğitim/benchmark datası için:

``` json
{
  "rating": "good",
  "difficulty_feedback": "too_easy"
}
```

Desteklenecek örnek değerler:

-   `good`
-   `bad`
-   `too_easy`
-   `correct`
-   `too_hard`

Feedback modeli daha sonra genişletilebilir.

------------------------------------------------------------------------

# 7. Generation Provider Interface

Bütün generation modelleri tek sözleşmeye uymalı.

Örnek abstraction:

``` python
from typing import Protocol

class GenerationProvider(Protocol):

    async def generate_text(
        self,
        *,
        model: str,
        messages: list,
        temperature: float,
        max_tokens: int | None = None,
    ) -> "GenerationResult":
        ...

    async def generate_structured(
        self,
        *,
        model: str,
        messages: list,
        schema: type,
        temperature: float,
    ) -> "GenerationResult":
        ...

    async def health_check(self) -> bool:
        ...
```

Normalize edilmiş sonuç:

``` python
class GenerationResult:
    content: str
    parsed: object | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int
    provider: str
    model: str
    raw_finish_reason: str | None
```

Application/domain kodu provider'ın raw response objesini görmemelidir.

------------------------------------------------------------------------

# 8. Embedding Provider Interface

Generation'dan ayrı abstraction oluştur.

``` python
class EmbeddingProvider(Protocol):

    async def embed(
        self,
        *,
        model: str,
        texts: list[str]
    ) -> "EmbeddingResult":
        ...

    async def health_check(self) -> bool:
        ...
```

Embedding modelinin değişmesi mevcut vektör indeksinin yeniden
oluşturulmasını gerektirebileceğinden şu metadata saklanmalıdır:

-   provider
-   model
-   embedding dimension
-   embedding version
-   indexed_at

Aynı vector collection/index içinde uyumsuz embedding modellerini
karıştırma.

------------------------------------------------------------------------

# 9. Provider Adapter Sistemi

İlk adapter:

``` text
OllamaGenerationProvider
OllamaEmbeddingProvider
```

Sonradan:

``` text
ProviderXGenerationProvider
ProviderYGenerationProvider
```

eklenebilmeli.

Yeni provider eklemek için:

1.  Provider interface implement et.
2.  Provider factory/DI container'a kaydet.
3.  Model Registry'ye model ekle.
4.  Routing policy'de aktif et.

Card generation kodunda değişiklik gerekmemeli.

------------------------------------------------------------------------

# 10. Capability Normalization

Bütün modeller aynı özellikleri desteklemez.

Capability modeli oluştur:

``` python
class ModelCapabilities:
    structured_output: bool
    json_mode: bool
    tool_calling: bool
    vision: bool
    streaming: bool
    max_context_tokens: int | None
```

`generate_structured()` çağrısında:

1.  Native schema/structured output destekleniyorsa onu kullan.
2.  JSON mode varsa JSON mode + parser kullan.
3.  Hiçbiri yoksa strict prompt + JSON parser kullan.
4.  Parse başarısızsa repair/retry mekanizmasını çalıştır.

Domain katmanı bu farklılıkları bilmemeli.

------------------------------------------------------------------------

# 11. Model Registry

PostgreSQL'de model registry oluştur.

Önerilen alanlar:

``` text
id
provider
model_name
model_type           generation | embedding
enabled
is_local
priority
input_cost_per_million
output_cost_per_million
currency
supports_structured_output
supports_json_mode
supports_tools
supports_vision
context_window
languages
quality_score
avg_latency_ms
health_status
created_at
updated_at
```

`quality_score` başlangıçta null olabilir.

Kendi benchmark sonuçlarımızdan güncellenmelidir.

Cloud fiyatlarını kod içine gömme; registry/config üzerinden yönet.

------------------------------------------------------------------------

# 12. Model Router

Kart servisi doğrudan provider seçmemeli.

Örnek:

``` python
result = await model_router.generate_structured(
    task="card_generation",
    messages=messages,
    schema=GeneratedCardBatch,
    requirements={
        "language": "tr",
        "structured_output": True
    }
)
```

Router aşağıdaki kriterleri destekleyecek şekilde tasarlanmalı:

-   enabled
-   health
-   capability
-   language
-   task
-   local/cloud
-   cost
-   quality
-   latency
-   priority

İlk MVP routing policy:

``` text
1. Primary model
2. Primary başarısızsa fallback #1
3. O da başarısızsa fallback #2
4. Tümü başarısızsa kontrollü service error
```

Retry sonsuz döngüye girmemeli.

------------------------------------------------------------------------

# 13. Routing Configuration

Örnek config:

``` yaml
routing:
  card_generation:
    providers:
      - provider: ollama
        model: qwen
        priority: 1

      - provider: cloud_x
        model: model-x
        priority: 2

    max_provider_attempts: 3
```

Development'ta yalnızca Ollama çalışabilmeli.

Cloud credentials yoksa uygulama startup sırasında gereksiz yere crash
etmemeli; ilgili provider disabled olmalı.

------------------------------------------------------------------------

# 14. RAG

RAG generation modelinden bağımsız olmalıdır.

Akış:

``` text
target/topic
    |
    v
query builder
    |
    v
EmbeddingProvider
    |
    v
pgvector similarity search
    |
    v
top-k context
    |
    v
Prompt Builder
    |
    v
Model Router
```

RAG doküman modeli:

``` text
id
language
word
definition
category
related_terms
source
source_license
metadata JSONB
embedding vector
embedding_model
embedding_version
```

Çoklu dil aramasında `language` filtresi uygula.

İlk sürümde `top_k` config üzerinden değiştirilebilir olsun.

RAG sonucu modele doğrudan güvenilir talimat olarak değil, `context`
olarak verilmelidir.

------------------------------------------------------------------------

# 15. Prompt Management

Promptları Python stringleri içine gömme.

Dosyalarda tut:

``` text
prompts/card/system_v1.md
prompts/card/generate_v1.md
prompts/card/repair_v1.md
```

Prompt metadata:

``` text
name
version
language strategy
created_at
```

Üretilen her kart batch'i ile kullanılan:

-   prompt name
-   prompt version

telemetry'de kaydedilsin.

Prompt içinde provider/model adı bulunmamalı.

------------------------------------------------------------------------

# 16. Structured Output

LLM'den beklenen örnek schema:

``` python
class GeneratedCard(BaseModel):
    target: str
    forbidden_words: list[str]
    category: str | None = None
    difficulty: Literal["easy", "medium", "hard"]

class GeneratedCardBatch(BaseModel):
    cards: list[GeneratedCard]
```

`count=10` istenirse validator tam 10 geçerli kart elde etmeye
çalışmalı.

Tek seferde aşırı büyük batch isteme. Config ile batch size belirle.

------------------------------------------------------------------------

# 17. Validation Pipeline

LLM çıktısı hiçbir zaman doğrudan database'e yazılmamalı.

Pipeline:

``` text
LLM output
   |
   v
Pydantic schema
   |
   v
Normalization
   |
   v
Rule validators
   |
   v
Duplicate checks
   |
   v
Semantic checks (opsiyonel)
   |
   +--> PASS --> persist
   |
   +--> FAIL --> repair/regenerate
```

İlk validator kuralları:

1.  Tam 5 forbidden word.
2.  Duplicate forbidden word yok.
3.  Target forbidden listesinde yok.
4.  Boş değer yok.
5.  Maksimum kelime uzunluğu.
6.  Dil karakterleri geçerli.
7.  Aynı batch içinde target tekrar etmiyor.
8.  Database'de aynı target + language + yakın forbidden kombinasyonu
    varsa duplicate adayı.
9.  Güvenlik/uygunsuz içerik filtresi genişletilebilir bir interface
    üzerinden uygulanabilmeli.

Validation sonucu:

``` python
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]
```

------------------------------------------------------------------------

# 18. Repair / Retry

Validation fail olduğunda tüm isteği körlemesine yeniden üretme.

Önce repair prompt kullanılabilir:

``` text
Bu kart aşağıdaki validation hatalarını aldı:
- forbidden_words 4 adet
- target yasaklı listede bulunuyor

Kartı aynı hedef kelimeyi koruyarak düzelt.
```

Maximum:

-   parse retry
-   validation repair retry
-   provider fallback

ayrı limitlere sahip olsun.

Limitler config üzerinden yönetilsin.

------------------------------------------------------------------------

# 19. Duplicate Detection

İki katman kullan:

## Exact

Normalize edilmiş:

``` text
language + target
```

ve forbidden word set kontrolü.

## Semantic

İleride embedding üzerinden kart/target benzerliği kontrol edilebilir.

Semantic similarity tek başına kartı reddetmemeli; configurable
threshold ile duplicate adayı olarak işaretlenmeli.

------------------------------------------------------------------------

# 20. Database Tabloları

Minimum tablolar:

## `cards`

``` text
id UUID
target
language
category
difficulty
source
status
created_at
updated_at
```

## `card_forbidden_words`

``` text
id
card_id
word
position
```

## `rag_documents`

``` text
id
language
word
content
category
source
source_license
metadata JSONB
embedding VECTOR
embedding_model
embedding_version
created_at
```

## `ai_models`

Model Registry.

## `ai_requests`

``` text
id
task
provider
model
prompt_name
prompt_version
input_tokens
output_tokens
estimated_cost
latency_ms
success
error_type
retry_count
created_at
```

## `card_feedback`

``` text
id
card_id
rating
difficulty_feedback
created_at
```

Kullanıcı kimliği eklenirse privacy-first tasarla ve gereksiz PII
saklama.

------------------------------------------------------------------------

# 21. Observability

Her AI request için ölç:

-   task
-   provider
-   model
-   local/cloud
-   latency
-   token kullanımı mevcutsa
-   estimated cost
-   HTTP/provider hata tipi
-   parse failure
-   validation failure
-   retry
-   fallback kullanıldı mı
-   prompt version
-   kart acceptance rate

API key veya prompt içindeki hassas verileri loglama.

------------------------------------------------------------------------

# 22. Cost Tracking

Cloud modeller için:

``` text
estimated_cost =
(input_tokens / 1_000_000 * input_price)
+
(output_tokens / 1_000_000 * output_price)
```

Fiyatlar registry'den gelsin.

Local model için API cost `0` olabilir; fakat bu gerçek altyapı
maliyetinin sıfır olduğu anlamına gelmediğinden alanı
`api_estimated_cost` gibi açık isimlendir.

------------------------------------------------------------------------

# 23. Health Check ve Circuit Breaker

Provider health:

``` text
healthy
degraded
unavailable
```

Sürekli hata veren provider'a her request göndermemek için circuit
breaker tasarımına uygun abstraction oluştur.

MVP'de basit failure counter kullanılabilir.

Endpoint:

``` http
GET /v1/health
```

Public health endpoint hassas provider detaylarını döndürmemeli.

Admin/debug endpoint ayrı tutulabilir.

------------------------------------------------------------------------

# 24. Çoklu Dil

Kart dili request ile gelir.

RAG araması aynı dilde yapılır.

Kartları bir dilden diğerine otomatik çevirerek ana üretim yöntemi
yapma.

Tercih edilen akış:

``` text
TR target -> TR RAG -> TR generation
EN target -> EN RAG -> EN generation
DE target -> DE RAG -> DE generation
```

İleride cross-lingual retrieval opsiyonel olarak eklenebilir.

Arapça/RTL konusu backend'de metni bozmadan Unicode olarak
saklanmalıdır; RTL presentation Flutter'ın sorumluluğudur.

------------------------------------------------------------------------

# 25. Local Development

Docker Compose en az:

``` text
api
postgres + pgvector
```

Ollama host makinede veya ayrı container olarak çalışabilir.

`.env.example`:

``` env
APP_ENV=development
DATABASE_URL=postgresql+asyncpg://...
DEFAULT_GENERATION_PROVIDER=ollama
DEFAULT_GENERATION_MODEL=...
DEFAULT_EMBEDDING_PROVIDER=ollama
DEFAULT_EMBEDDING_MODEL=...
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

Model adlarını hard-code etme.

------------------------------------------------------------------------

# 26. Güvenlik

-   API keys `.env`/secret manager üzerinden.
-   Repository'ye secret commit etme.
-   Request validation.
-   Rate limiting için interface/extension point bırak.
-   Provider error body'lerini doğrudan client'a döndürme.
-   SQL injection'a karşı ORM/parameterized query.
-   Prompt injection'a karşı RAG context ile system instruction ayrımını
    koru.
-   RAG kaynak metnini executable instruction gibi değerlendirme.
-   Maksimum input uzunlukları belirle.
-   Admin endpointlerini public API'den ayır.

------------------------------------------------------------------------

# 27. Benchmark Sistemi

`tests/benchmark/` altında model karşılaştırma altyapısı oluştur.

Benchmark dataset örneği:

``` json
[
  {
    "target": "uçak",
    "language": "tr",
    "difficulty": "medium"
  },
  {
    "target": "kahve",
    "language": "tr",
    "difficulty": "medium"
  }
]
```

Her model için ölç:

-   JSON/schema success rate
-   validation pass rate
-   duplicate rate
-   average latency
-   token usage
-   estimated API cost
-   human rating (sonradan)
-   retry rate

Benchmark sonucundan model registry `quality_score` üretmek için
extension point bırak.

İlk sürümde otomatik quality score uydurma.

------------------------------------------------------------------------

# 28. Test Stratejisi

Unit tests:

-   Card validator
-   normalization
-   routing
-   fallback
-   cost calculation
-   capability selection
-   prompt rendering

Integration tests:

-   PostgreSQL
-   pgvector
-   Ollama adapter
-   `/cards/generate`

Provider testlerinde gerçek cloud API çağrıları default test suite
içinde zorunlu olmamalı.

Mock/Fake provider oluştur:

``` python
class FakeGenerationProvider:
    ...
```

Böylece application testleri LLM olmadan çalışabilsin.

------------------------------------------------------------------------

# 29. Hata Modeli

Normalize edilmiş internal exceptions:

``` text
ProviderUnavailable
ProviderTimeout
RateLimitExceeded
StructuredOutputError
ValidationExhausted
NoEligibleModel
EmbeddingError
RAGUnavailable
```

FastAPI bunları kontrollü public error response'a dönüştürsün.

Provider-specific exception'lar application katmanına sızmamalı.

------------------------------------------------------------------------

# 30. İlk Uygulama Sırası

AI coding agent aşağıdaki sırayla geliştirsin:

### Phase 1 --- Skeleton

-   FastAPI
-   settings
-   PostgreSQL
-   Alembic
-   health endpoint
-   clean architecture klasörleri

### Phase 2 --- AI Abstraction

-   GenerationProvider
-   EmbeddingProvider
-   normalized result types
-   capabilities
-   FakeProvider

### Phase 3 --- Ollama

-   Ollama generation adapter
-   Ollama embedding adapter
-   health check

### Phase 4 --- Cards

-   domain models
-   Pydantic DTO
-   validator
-   CardRepository
-   `/v1/cards/generate`

### Phase 5 --- RAG

-   pgvector
-   document ingestion
-   embedding
-   retrieval
-   prompt context

### Phase 6 --- Router

-   Model Registry
-   primary/fallback
-   capability filtering
-   provider health

### Phase 7 --- Observability

-   ai_requests
-   latency
-   token
-   cost
-   retry/fallback metrics

### Phase 8 --- Benchmark

-   benchmark dataset
-   runner
-   report JSON/CSV

Cloud provider adapterları bundan sonra eklenebilir.

------------------------------------------------------------------------

# 31. Definition of Done --- MVP

MVP tamamlanmış sayılması için:

-   `docker compose up` ile backend/database kalkmalı.
-   Ollama ile local generation çalışmalı.
-   Ollama unavailable olduğunda kontrollü hata dönmeli.
-   `/v1/cards/generate` valid kart üretmeli.
-   RAG enable/disable config ile açılıp kapanabilmeli.
-   pgvector retrieval çalışmalı.
-   Card validator geçersiz LLM çıktısını database'e sokmamalı.
-   Promptlar versiyonlu dosyalarda bulunmalı.
-   Model adı domain/application içine hard-code edilmemeli.
-   Generation ve embedding providerları ayrı olmalı.
-   Fake provider ile unit testler LLM olmadan çalışmalı.
-   Model Registry provider/model değiştirmeyi desteklemeli.
-   Fallback en az iki provider adapterı varmış gibi test edilebilmeli.
-   AI request telemetry database'e yazılmalı.
-   README kurulum ve local test adımlarını içermeli.
-   `.env.example` bulunmalı.
-   API OpenAPI/Swagger üzerinden test edilebilmeli.

------------------------------------------------------------------------

# 32. Kritik Mimari Yasaklar

Aşağıdakileri YAPMA:

``` python
# Card service içinde:
ollama.chat(...)
```

YAPMA.

``` python
# Flutter request:
{"model": "qwen"}
```

YAPMA.

``` python
# Domain:
from openai import ...
```

YAPMA.

``` python
# Aynı abstraction:
class AIProvider:
    chat()
    embed()
    postgres()
    ...
```

YAPMA.

Bunun yerine küçük ve ayrı interface'ler kullan.

------------------------------------------------------------------------

# 33. Temel Tasarım Prensibi

Bu backend'in gelecekte şu değişiklikleri minimum kod değişimiyle
kaldırması gerekir:

``` text
Bugün:
Ollama + Qwen local

Yarın:
Ucuz Cloud Provider A

Daha sonra:
Cloud Provider B + yeni model

Fallback:
Local Qwen

Embedding:
Sabit multilingual embedding model

Daha sonra:
Yeni embedding modeli + yeni vector index version
```

Kart üretme domain mantığı bu değişikliklerden etkilenmemelidir.

------------------------------------------------------------------------

# 34. AI Coding Agent İçin Son Talimat

Bu dokümanı sadece mimari öneri olarak açıklama.

**Çalışan backend projesini oluştur.**

Kurallar:

1.  Önce klasör yapısını ve implementation planını çıkar.
2.  Sonra gerçek dosyaları oluştur.
3.  Placeholder/pseudo-code bırakmamaya çalış.
4.  Her phase sonunda test çalıştır.
5.  Provider-specific kodu infrastructure dışında kullanma.
6.  Domain/application katmanlarını provider SDK'larından bağımsız tut.
7.  Local Ollama ilk çalışan provider olsun.
8.  Cloud provider eklemek için açık extension point bırak.
9.  Public Flutter API'sini provider/model değişimlerinden bağımsız tut.
10. README içinde:

-   kurulum
-   Docker
-   PostgreSQL/pgvector
-   Ollama
-   migration
-   test
-   örnek API request adımlarını eksiksiz yaz.

11. Kararsız kaldığın küçük implementation detaylarında bu mimari
    ilkelerine uygun makul varsayım yap; projeyi yarım bırakma.
12. Güvenlik anahtarlarını veya gerçek secret değerlerini repository'ye
    yazma.

İlk hedef: **local Ollama ile çalışan, RAG destekli, provider-agnostic
kart üretme MVP backend'i.**
