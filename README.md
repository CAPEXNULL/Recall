# Recall

An AI agent that learns from the outcomes of its own decisions and reuses that experience to handle previously unseen situations.

Built as a new, independent entry for the **Nebius x NVIDIA Global AI Hackathon — Best Apps and Agents Track**.

## Problem and behavior

Small operations teams repeatedly encounter similar service incidents. A one-off assistant forgets whether its previous suggestion helped. Recall links a suggested action to a reported outcome, then shows exactly which successful experience influenced a later recommendation.

`Situation → Decision → Outcome → Experience → Transfer → New Decision`

1. Describe a situation. Without matching successful experience, Recall asks an NVIDIA Nemotron model through Nebius Token Factory for an action and explanation.
2. Try the action yourself and report success, a score from 0 to 1, and feedback.
3. Describe a new situation. If confirmed experience matches, Recall reuses the action without another model request and displays its source, similarity and score.

Actions are suggestions; Recall does not execute infrastructure commands. Reported outcomes are user observations, not independently verified measurements. Learning means updating an experience store, not training model weights.

## Architecture

- `app.py`: one Russian/bilingual screen, Flask forms and isolated server-side browser sessions.
- `recall_core.py`: deterministic state changes, outcome linkage, experience selection and provenance.
- `model_client.py`: standard-library HTTPS adapter for Nebius Token Factory; strict response validation; no automatic retries.
- `demo/scenario.json`: one example linking incidents on two different services.
- `tests/test_recall.py`: local core, transfer, adapter and real-browser checks, plus an explicitly enabled live check.

Memory remains in one server process. Reloading retains it; a server restart or 24 hours without activity clears it. Sessions have a limit of 100 decisions. No database or background worker is needed for this demonstration.

## Matching contract

Features are Unicode alphanumeric words/numbers, normalized with NFC and case-folding, deduplicated and sorted. The model returns the same canonical feature set with its action and reason. This permits local matching before deciding whether a model call is needed.

Similarity is Jaccard: `|A ∩ B| / |A ∪ B|` (zero for an empty union). An experience is eligible only when success is true, its score is at least **0.50**, and similarity is at least **0.50**. Ties resolve by higher similarity, higher score, lexicographically smaller NFC action, then smaller source experience ID.

Inputs: situation 1–2000 characters and at most 200 distinct features; feedback up to 1000 characters; finite score in [0,1]. A decision accepts one immutable outcome. Invalid inputs or invalid model responses leave state unchanged. Each browser has its own memory; one-time form tokens prevent replay and duplicate submissions.

## Nebius and NVIDIA integration status

Endpoint: `https://api.tokenfactory.nebius.com/v1/chat/completions`.

Required model family: NVIDIA Nemotron. **Verified and locked model ID: `nvidia/Nemotron-3_5-Lightning`.**

The authenticated account model list includes Lightning, Nano, Super and Ultra. Lightning is a compact choice at 3B active parameters, tied with Nano for the smallest active size in that list. Its model ID and JSON schema were fixed before the live request. One real request passed: HTTP 200, valid structured output, and a MODEL decision saved and rendered by the application. The measured application check took 5.329 seconds and used 897 tokens (177 input, 720 output). These are observations from one request, not a performance benchmark. The model lock and evidence remain outside the public directory. Do not change the locked model unless incompatibility or unavailability is demonstrated.

Nebius supplies inference for unmatched situations. NVIDIA Nemotron produces structured `features`, `action`, and `reason`. The adapter requests `response_format: json_schema` and validates the response before the application records it. A transfer uses no model call.

Sources checked 2026-10-01: [competition requirements](https://nebiusglobalaihackathon.devpost.com/rules), [official model catalog](https://github.com/nebius/token-factory-cookbook/blob/main/models/nemotron/README.md), [structured outputs](https://docs.tokenfactory.nebius.com/ai-models-inference/json).

## Setup

Python 3.11 or later. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Provide the following environment variables through the terminal or hosting secret manager:

| Variable | Purpose |
| --- | --- |
| `NEBIUS_API_KEY` | Token Factory API credential; server-only |
| `NEBIUS_MODEL_ID` | Exact verified NVIDIA Nemotron model ID |
| `RECALL_SESSION_SECRET` | Optional local / recommended hosted signing secret; otherwise generated per process |
| `RECALL_HTTPS` | Set to `1` only when the browser uses HTTPS, enabling secure cookies |

`.env.example` contains names only. The application does **not** automatically read `.env`. Do not paste credentials into source files, screenshots, test evidence or chat. To enter an API key in PowerShell without displaying it:

```powershell
$recallCredential = Read-Host 'Nebius API key' -AsSecureString
$env:NEBIUS_API_KEY = [System.Net.NetworkCredential]::new('', $recallCredential).Password
```

Set `NEBIUS_MODEL_ID` to the verified ID, then launch using the virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
$env:NEBIUS_MODEL_ID = 'nvidia/Nemotron-3_5-Lightning'
python app.py
```

Open **http://127.0.0.1:5000**. With no credentials the screen remains usable and clearly reports the missing connection; it never silently substitutes a fake model.

## Demonstration

The first situation is prefilled from `demo/scenario.json`: service Alpha became slow after an update and a rollback is available.

1. Click **Получить решение / Get decision**. The initial source must be **Модель / MODEL**.
2. For this simulated incident, mark **Успех / Success**, enter score **0.8**, describe the observed outcome, and click **Сохранить исход / Save outcome**.
3. Replace **Альфа** with **Бета** in the situation and request another decision.
4. The source becomes **Опыт / EXPERIENCE**. Verify the original experience ID, similarity, score and reused action in **Доказательство переноса / Transfer evidence**.

The demo is intentionally a simulated incident; it does not claim to have modified a real service. The new text has a different service name and was not previously in the session history.

## Tests

For browser verification install the test-only dependency and Chromium:

```powershell
python -m pip install playwright==1.52.0 pathspec==0.12.1
python -m playwright install chromium
python tests/test_recall.py
```

Default execution checks local behavior using an explicitly labelled test double; TEST-03 is skipped and remains unverified. Local success does not establish real Nebius integration or final acceptance. Tests stop on the first failure. Test artifacts, screenshots and TRACE default to `../evidence/<run_id>/`, outside this public directory. Configured artifact paths inside `recall/` are rejected before any test runs. The test-only `pathspec` dependency checks actual `.gitignore` matching without Git commands; it is not needed to run the application.

For a complete acceptance run, first freeze a private test-plan manifest and a model-lock JSON containing `model_id`, the account's actual `available_models` list and `json_mode_supported: true` with supporting evidence. Configure `RECALL_MODEL_LOCK` with its path and `RECALL_FULL_RUN=1`. The ordered run then executes TEST-01..04, with exactly one live inference in TEST-03. Do not repeat a clean full pass; rerun only after a documented defect fix changes the tested version. Optional `RECALL_TRACE_PATH`, `RECALL_EVIDENCE_DIR` and `RECALL_RUN_ID` direct evidence outside the public directory.

## Deployment instructions

Publication is performed separately by the project owner. Use a Python host with HTTPS termination, install `requirements.txt`, and supply secrets through its environment settings. Start **one** process:

```text
waitress-serve --host=0.0.0.0 --port=8080 --call app:create_app
```

Route the host's HTTPS endpoint to port 8080 and set `RECALL_HTTPS=1`. Keep one process/instance so in-memory sessions stay consistent. A restart clears memory; a judge can replay the short demonstration. Set provider-side spending limits for a publicly accessible model-backed demo. Confirm the hosted A → outcome → B path before submitting its URL.

## Video script — 2 minutes 30 seconds

Use English narration or English subtitles for the Russian screen.

| Time | Screen and narration |
| --- | --- |
| 0:00–0:20 | “Recall remembers whether an action worked and brings that evidence to the next decision.” Show the screen and explain the incident-response use case. |
| 0:20–0:55 | Enter Alpha's situation. Show the actual Nemotron response via Nebius Token Factory and the MODEL label. |
| 0:55–1:20 | Explain that this is a simulated incident. Save the successful reported outcome, score 0.8 and feedback. Show the linked experience. |
| 1:20–1:55 | Change Alpha to Beta. Show the EXPERIENCE decision and explain that the new situation reuses the action without another inference request. |
| 1:55–2:20 | Show the source record, similarity and outcome score. Explain the lexical matching limitation and human review. |
| 2:20–2:30 | “Nebius hosts NVIDIA Nemotron inference; Recall adds outcome-linked memory and visible evidence.” Show repository and demo URLs after publication. |

## Limitations

Lexical overlap is not semantic understanding: synonyms, negation, service-specific constraints and changed circumstances can make a retrieved action inappropriate. Users must review actions, including service names that may be carried over from the source decision. Failed outcomes are retained but not used for positive transfer. An old successful outcome is not automatically invalidated by a later failure. Memory is volatile; no cross-session or long-term persistence is claimed. The configured model passed one live schema/application check; broader reliability and recommendation quality are not established by that single result.

## Technology feedback

- **Used / purpose:** Nebius Token Factory ran `nvidia/Nemotron-3_5-Lightning` for the initial structured incident recommendation. Local core and browser scenarios use a test double.
- **Worked well locally:** deterministic matching and visible provenance passed the local checks. In the live check, the first response satisfied the JSON schema and exact canonical feature validation, and the app saved the returned action and reason without a retry.
- **Difficulty:** the key initially existed outside the agent's process environment; loading it from the Windows user environment resolved access. Account model-list verification was needed before choosing the compact model.
- **Onboarding:** authenticated model discovery succeeded, followed by one successful inference. No additional inference requests were used to compare models.
- **Suitability:** the observed request supports this demonstration's integration contract. It does not establish production reliability or independently verify that the suggested infrastructure action would succeed.

## Submission status

`SUBMISSION_READY = false`

The owner must choose an open-source license (`LICENSE_REQUIRES_USER_APPROVAL`), publish the repository and working demo, upload a public YouTube video under three minutes, finish Devpost registration, and submit the entry. Repository URL, demo URL and YouTube URL are `REQUIRES_USER_ACTION`. Technical readiness and contest submission readiness are separate.
