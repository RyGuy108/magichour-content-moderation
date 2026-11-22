# Content moderation pipeline for generative media

A moderation layer that sits in front of and behind the Magic Hour API. It **pre-checks** prompts and input
images before a render is paid for, **post-checks** generated images and videos before they reach the end
user, and sends uncertain cases to a **human review queue**. It flags sexual content, sexual content involving
minors, gore, hate, self-harm, real-person likeness (deepfakes), and trademarks/logos. Every decision is stored
with scores and evidence. Policy thresholds are configured per tenant, so changing a policy never needs a retrain.

On a hand-written test set the model never saw, it catches **80% of unsafe prompts where a keyword blocklist
catches 57%**, with **half the false positives** (6.9% vs 13.8%). On obfuscated prompts (`n.u.d.e`, `t@yl0r`,
Cyrillic look-alike letters, zero-width spaces) it catches **94% vs 27%**. The prompt check adds **3.9 ms p50**
end to end over HTTP, including the Postgres write. For video, the frame sampler catches **93% of 0.25-second
inserted shots while checking 4.9 frames per clip**. Uniform 1 fps sampling catches 20%, and checking every frame
costs 11× more. In a live run against Magic Hour, the pre-check stopped **3/3 face swaps of protected people
and 2/2 abusive prompts before any credits were spent**. Reviewer decisions flow back into the prompt model
through a gated retraining loop, and every client authenticates with a tenant API key that is rate-limited in Redis.

**Stack:** Python 3.13, FastAPI, PyTorch, OpenCV, onnxruntime, dlib, Postgres 16, Redis + RQ, Next.js/TypeScript,
Docker Compose, GitHub Actions. 34 tests, run in CI against real Postgres and Redis.

![results](results/summary.png)

## Results

All numbers come from `results/*.json`, which the scripts in `bench/` regenerate.

### Prompt pre-check (`bench/eval_text.py`)

"Unsafe" means the prompt has any policy label. Our pipeline flags a prompt when its decision is review or
block. The baseline is a keyword blocklist that gets the same term lists, the same 170 celebrity names, the same
150 brands and characters, and a minor + sexual co-occurrence rule (`bench/baseline.py`). The comparison is about
method, not about who has the longer list.

| Set | n | Recall: blocklist → ours | Precision: blocklist → ours | False-positive rate: blocklist → ours |
|---|---|---|---|---|
| **Hand-written, final** (`natural_v2.tsv`, written after the model was frozen, scored once) | 141 | 0.57 → **0.80** | 0.85 → **0.94** | 13.8% → **6.9%** |
| Obfuscated (test positives rewritten 4 ways, plus obfuscated benign prompts) | 2,600 | 0.27 → **0.94** | 1.00 → 1.00 | 0.0% → 3.5% |
| Held-out templates + held-out names + 1,500 COCO captions | 2,637 | 0.67 → **0.95** | 0.81 → **0.93** | 4.5% → **2.2%** |
| Hard negatives only ("bloody mary", "naked mole rat", "apple pie", suicide-prevention poster...) | 328 | n/a | n/a | 15.5% → 13.7% |

- **Latency:** model + policy take 0.37 ms p50 and 0.84 ms p99 on one CPU thread. Over HTTP with the Postgres
  insert, one client sees 3.9 ms p50 and 7.1 ms p99. Four API processes sustain 383 req/s at 32 concurrent
  clients on a 4-core box that is also running the load generator and Postgres (`results/load.json`).
- `natural.tsv` was used as a dev set while building, so its numbers (recall 0.88) are optimistic. Quote the
  `natural_v2` numbers.
- **Where it still misses:** euphemisms with no risky words ("a woman lying in bed wearing nothing at all", "doing it
  doggy style"). It also misses real people the gazetteer doesn't know ("zuckerberg", short names like "trump and
  putin kissing") and marks named only by a nickname ("ferrari prancing horse"). Per-label numbers and every
  error are in `results/text.json`.

### Image detectors (`bench/eval_images.py`, Magic Hour-generated test images)

The images are synthetic and generated with the real Magic Hour API (`bench/make_images.py`, ledger of every paid
render in `data/gen/ledger.jsonl`). The test set uses no real person's face and no real company's logo.
Thresholds are picked on the dev half and reported on the test half.

| Detector | Test set | Result |
|---|---|---|
| **NSFW** (NudeNet YOLOv8n, ONNX) | 30 hard negatives: swimwear, gym, bodybuilding, statues, ballet, lingerie flat-lay | **0 false positives** (max score 0.38 vs 0.45 review threshold), 23 ms/image. Recall isn't measured here, because the repo deliberately contains no explicit imagery. |
| **Protected face** (YuNet detector, SFace + dlib ResNet embeddings, fused) | 10 enrolled identities, one photo each. 30 Magic Hour face swaps of them into new scenes, vs 50 photos of other people + 30 swaps of non-enrolled "impostor" faces | ROC AUC **0.92** (v1, dlib only: 0.86). **47% recall with 0 false matches** at the review threshold (v1: 20% at its best zero-false-match cutoff), 73% at a 5% false-match rate. 0 false matches on the 39-image "default face" stress set (v1: 38 of 40 at dlib's default cutoff). 230 ms/image. |
| **Registered logo** (SIFT/RootSIFT retrieval over frontal + simulated off-axis views, similarity-RANSAC verification) | 8 invented marks printed onto 30 Magic Hour product/street scenes with perspective, shading, blur and JPEG damage, vs clean scenes and 4 look-alike unregistered marks | **57% recall at 3% FPR** (v1: 40%), 50% at 0% (v1: 33%). 0 hits on look-alike marks. 390 ms/image. |

**What moved the two weak detectors.** Faces: dlib and SFace make mostly independent mistakes on generated
faces, so the matcher scores `cos(SFace) - 0.5 * dist(dlib)`. SFace alone reaches AUC 0.89 and dlib 0.87;
the fusion reaches 0.92 (`results/faces_v1_dlib.json` keeps the old numbers). Logos: SIFT is only invariant to
rotation and scale, but a print on a hanging shirt is a perspective view. Indexing four simulated tilts of each
mark (the ASIFT idea) and verifying all eight marks instead of the top three accounts for the whole gain.
Things that did not help: jittered or flipped face embeddings, augmented enrollment photos, CLAHE, lower SIFT
contrast thresholds, full-affine RANSAC (more false hits), and multi-scale references.

**What I learned building the face set:** Magic Hour's `z-image-turbo` renders "a person ..." as nearly the same
young East Asian man every time. The median dlib distance between those supposedly different people is 0.47,
under dlib's own 0.6 same-person cutoff. One of the protected identities fit that description, so 38 of 40
"default person" images false-matched him at dlib's default threshold (`default_face_stress_set` in
`results/images.json`). Negatives now describe who is in the picture. The general lesson is that face matching
on AI-generated people needs a stronger embedding than dlib, which is what v2 added.

### Video post-check (`bench/eval_video.py`)

The benchmark uses 16 real Magic Hour clips generated earlier in this project. A 0.25 s, 0.5 s or 1 s shot of a
protected person is spliced into a random point of each clip, and the question is whether the post-check catches it.

| Strategy | Frames checked per clip | Caught 0.25 s | Caught 0.5 s | Caught 1 s | ms per clip |
|---|---|---|---|---|---|
| Every frame | 121 | 100% | 100% | 100% | 8,551 |
| Uniform 1 fps | 6.0 | 20% | 46% | 100% | 976 |
| Uniform 2 fps | 11.0 | 40% | 100% | 100% | 1,227 |
| **modpipe sampler** (scene-change trigger + 1 s max gap + dHash dedupe) | **4.9** | **93%** | **100%** | **100%** | **805** |

This table was measured with the v1 (dlib) face model, which counts an insert only if the model recognizes the
inserted still; that gave 43 inserts. With v1, the clean clips produced 1/16 false alarms with the sampler and
3/16 when checking every frame. Re-running with the v2 face model (`results/video_faces_v2.json`) removes every
false alarm on the clean clips (0/16 for all strategies), but v2 recognizes fewer of the inserted stills at its
stricter review threshold, so only 19 inserts qualify. On those, the sampler catches 6/6 quarter-second, 5/7
half-second and 6/6 one-second inserts, versus 1/6, 2/7 and 6/6 at 1 fps. The v2 face model also makes each
checked frame slower (1.3 s per clip with the sampler).

### Live run against Magic Hour (`bench/demo_live.py`, `results/live.json`)

The run used the real API through `ModeratedMagicHour` (`modpipe/gateway.py`), with the 10 synthetic identities
enrolled. Five of ten requests were blocked at pre-check, which saved 40 credits. Three were attempts to face-swap
a protected person, blocked in ~0.5 s before upload. The other two were the obfuscated celebrity-nude prompt and a
hate poster. "nike logo on a hoodie" rendered but was held for review. Three benign prompts and one ordinary swap
went through, and their outputs passed post-check in 140-280 ms. The whole project spent about 2,100 Magic Hour
credits, nearly all on building the image test set.

### Review queue (`tests/test_review_queue.py`)

24 concurrent reviewers drained 400 items with 0 double claims, at ~700 claim+resolve/s. Items involving minors
jump the queue. A reviewer who walks away loses their lease, the item returns to the pool, and their late
"resolve" is rejected with 409.

## Architecture

```
            prompt + input images                          generated image / video
 client ──▶ POST /v1/precheck (sync, ~4 ms text-only) ... Magic Hour render ... POST /v1/postcheck (202)
                 │                                                                     │ RQ queue (Redis)
                 ▼                                                                     ▼
        ┌──────────────────── Checker (one per process, models loaded once) ───────────────────────┐
        │ normalize → gazetteer → PyTorch n-gram classifier    NudeNet ONNX (batched)               │
        │ policy.yaml per tenant → allow / review / block      YuNet → SFace + dlib → id gallery    │
        │                                                      SIFT index → logo verification       │
        │                                                      video: scene-aware frame sampler,    │
        │                                                      early exit on first blocking frame   │
        └──────────────────────────────────────────────────────────────────────────────────────────┘
                 │ every decision + scores + evidence
                 ▼
        Postgres: checks · review_items (SKIP LOCKED leases, priority) · protected_identities ·
                  identity_embeddings · logo_refs · api_keys · audit_log · registry_version (LISTEN/NOTIFY)
                 ▲                                         Redis: decision cache (content hash +
                 │                                                policy + model + registry version)
        Next.js review console (reviewer/) — claim, see evidence boxes, allow/block, correct labels
```

- **Policy lives apart from the models.** Detectors only output scores, and `policy.yaml` maps them to actions
  per tenant. An adult platform can allow nudity, and a licensed merch shop can allow its own logos. Tenants
  can never loosen `minor_sexual`, `csam_risk` or `real_person_sexual`.
- **Derived labels.** A real person combined with sexual content is non-consensual intimate imagery and is
  blocked, while likeness alone goes to review. Any minor signal combined with sexual content is treated as CSAM
  risk at the top of the queue.
- **Obfuscation is a signal.** If the normalizer had to undo leetspeak, homoglyphs or spacing, a borderline
  prompt is escalated to review instead of allowed.
- **Registries reload live.** Enrolling a face or registering a logo bumps `registry_version` and fires
  `NOTIFY`. Every API and worker process reloads its in-memory gallery without a restart, and cache keys include
  the version, so stale decisions can't be served.
- **Reviewer labels become training data.** See "Retraining loop" below.
- **Tenant API keys.** The key, not the request body, decides the tenant, so a client can't pick a looser
  tenant's policy. Roles: `client` (checks), `reviewer` (queue, acting as the key's name), `admin` (enrollment).
  Keys are stored as SHA-256 only and minted with `python -m modpipe.auth create`. Each key has a token bucket
  in Redis, updated by one Lua script so it's atomic across API processes (429 + `Retry-After`; fails open if
  Redis is down). The console's images and videos load through short-lived HMAC-signed URLs, because `<img>`
  tags can't send an `Authorization` header.

## Retraining loop (`modpipe/feedback.py`, `bench/retrain.py`, `bench/feedback_loop.py`)

Reviewers only ever see what the model flagged, so a loop built on the queue alone learns from false positives
and never from misses. A random **audit sample** of allowed checks (`AUDIT_SAMPLE_RATE`, default 1%) therefore
also enters the queue, at the lowest priority. `python -m modpipe.feedback export` turns resolved prompt reviews
into training rows (majority vote per normalized prompt, ties dropped). `bench/retrain.py` trains a candidate
with them and promotes it only if test recall drops by at most 1 point and hard-negative false positives rise by
at most 1 point. The new model version is part of the cache key, so old decisions stop being served.

`bench/feedback_loop.py` runs the whole loop through the real queue: the 203 hand-written dev prompts arrive as
traffic, simulated reviewers resolve what gets queued, and the model is retrained from the same seed with and
without the feedback. Scored on `natural_v2`, which is never trained or gated on (`results/feedback.json`):

| Audit rate | Reviewed rows | natural_v2 recall | natural_v2 FPR | Gate |
|---|---|---|---|---|
| no feedback | 0 | 0.795 | 6.9% | |
| 25% | 61 | **0.807** | 6.9% | pass |
| 100% | 133 | 0.819 | 8.6% | **rejected**: hard-negative FPR 17.3% → 19.0% |

The gains are small, which is what 61 examples against 15k templated rows should give. The point is that the
plumbing works end to end and that the gate catches a candidate that trades precision for recall.

## How the prompt model works

1. **Normalize:** NFKC, then strip zero-width characters and diacritics, map Cyrillic/Greek/Armenian look-alikes
   to Latin, collapse `n.u.d.e` / `N U D E`, undo leetspeak in letter-heavy tokens (`4k`, `1080p` and `$20` are
   left alone), and squash repeated letters.
2. **Gazetteer:** names, brands and characters become placeholders (`<person>`, `<brand>`, `<brand_ambig>` for
   "apple", "target", "jaguar"...). The model learns context ("`<person>` topless", "`<brand>` logo on a hoodie")
   instead of memorizing names, so adding a protected name is a list edit, not a retrain. Blocklist hits on
   the normalized text become `<lex_*>` features.
3. **Model:** a fastText-style PyTorch net. Four mean-pooled EmbeddingBags (special tokens, words, bigrams, char
   3-5-grams) share one 2^20 hashed table, followed by an MLP and 7 sigmoids. It trains in about a minute on CPU.
   Only rows touched in training ship (fp16), so the model is 0.9 MB instead of 200 MB.
4. **Training data** (`bench/make_prompts.py`): compositional templates, plus 8,000 MS-COCO captions as diverse
   safe prompts, plus 3,000 COCO captions with harmful phrases injected. Without the injected captions, the
   model learned that "everyday words = safe". Bag-level feature dropout (whole word/char bags dropped, special
   tokens kept) forces the model to act on `<person>` or `<lex_sexual>` when the wording is new.

The test split holds out phrasing (templates, wrappers, context fragments, hard-negative sentences) and 20% of
names and brands. The hand-written sets hold out vocabulary.

## Running it

```bash
scripts/fetch_data.sh                 # dlib, YuNet, SFace and res10 weights, COCO captions
docker compose up --build             # api :8000, 2 workers, review console :3000
docker compose exec api python -m modpipe.auth create --tenant default --name gateway          # client key
docker compose exec api python -m modpipe.auth create --tenant '*' --name you --role reviewer  # console key
docker compose exec api python -m modpipe.auth create --tenant '*' --name ops --role admin     # enrollment

# or locally (Postgres 16 + Redis running):
pip install -r requirements.txt
uvicorn modpipe.api:app --port 8000 &
WORKERS=2 python -m modpipe.worker &
cd reviewer && npm install && npm run dev

curl -s localhost:8000/v1/precheck -H "Authorization: Bearer $KEY" -H 'content-type: application/json' \
  -d '{"prompt": "t@yl0r sw1ft n.u.d.e on a beach"}'
# {"action": "block", "reasons": [{"label": "sexual", ...}, {"label": "real_person_sexual", ...}], "flags": ["leet", "spaced"], ...}

curl -s localhost:8000/v1/identities -H "Authorization: Bearer $ADMIN_KEY" -F identity_id=talent_42 -F name="Jane Doe" -F photos=@jane.jpg
curl -s localhost:8000/v1/logos -H "Authorization: Bearer $ADMIN_KEY" -F logo_id=acme -F name=Acme -F image=@acme.png
```

Reproducing the numbers:

```bash
python bench/make_prompts.py && python bench/train_text.py && python bench/eval_text.py
python bench/make_images.py all && python bench/make_images.py round2 && python bench/make_images.py negatives  # ~2,000 credits
python bench/make_logo_set.py && python bench/eval_images.py && python bench/eval_video.py
python bench/load_precheck.py 3000 32 && python bench/demo_live.py && python bench/plot.py
python bench/feedback_loop.py                                    # retraining loop experiment
pytest -q
```

The generated images (~500 MB) aren't committed. `data/gen/ledger.jsonl` lists every Magic Hour project id.

## Limitations and next steps

- **Faces.** Half of the protected-face swaps still get through at zero false matches, with one enrollment
  photo per person. Multi-photo enrollment (the API already accepts several) and a larger ArcFace model are the
  next levers. Existing 128-d enrollments from v1 must be re-enrolled (migration 002 keeps them but the matcher
  skips them).
- **Logos.** Keypoint matching is near its ceiling on small, low-texture marks. A small learned detector trained
  on composites from `bench/make_logo_set.py` is the obvious next step.
- **The live run and the load test** were recorded with the v1 face model and before API keys existed.
- **NSFW recall** needs a licensed labeled set, which is out of scope for a public repo. Image-side CSAM needs
  hash matching (PhotoDNA / NCMEC), which is also out of scope. Here, minors are handled at the prompt.
- The prompt model's training data is synthetic. The retraining loop is how real traffic would replace it.

Data credits: MS-COCO captions (CC BY 4.0), NudeNet (AGPL-3.0 model weights, loaded at runtime from the `nudenet`
package), dlib models (public domain / CC0), OpenCV res10, YuNet and SFace models (OpenCV model zoo, Apache-2.0 / MIT).
