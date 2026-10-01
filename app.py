"""One-screen Recall demo. Each browser has isolated, volatile server memory."""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import secrets
from threading import Lock
import time

from flask import Flask, render_template_string, request, session, redirect, url_for
from model_client import NebiusClient, ModelError
from recall_core import Recall

SCENARIO = json.loads((Path(__file__).parent / 'demo/scenario.json').read_text(encoding='utf-8'))


@dataclass
class BrowserState:
    core: Recall
    nonce: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    touched: float = field(default_factory=time.monotonic)
    lock: Lock = field(default_factory=Lock)


def create_app(model=None):
    app = Flask(__name__)
    app.config.update(SECRET_KEY=os.environ.get('RECALL_SESSION_SECRET') or secrets.token_hex(32),
                      MAX_CONTENT_LENGTH=16_384, SESSION_COOKIE_HTTPONLY=True,
                      SESSION_COOKIE_SAMESITE='Lax',
                      SESSION_COOKIE_SECURE=os.environ.get('RECALL_HTTPS') == '1')
    states, states_lock = {}, Lock()
    app.extensions['recall_states'] = states

    @app.after_request
    def headers(response):
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Content-Security-Policy'] = "default-src 'self'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
        return response

    @app.route('/', methods=['GET', 'POST'])
    def index():
        with states_lock:
            current = time.monotonic()
            for key in list(states):
                if current - states[key].touched > 86400 and not states[key].lock.locked():
                    del states[key]
            sid = session.get('sid')
            if sid not in states:
                if len(states) >= 1000:
                    return 'The demo is temporarily busy. Please try again later.', 503
                sid = secrets.token_urlsafe(24)
                session['sid'] = sid
                states[sid] = BrowserState(Recall(model if model is not None else NebiusClient()))
            state = states[sid]
            state.touched = current
        with state.lock:
            error = ''
            values = dict(situation=SCENARIO['situation_a'], success=False, score='0.8', feedback='')
            if request.method == 'POST':
                values.update(situation=request.form.get('situation', ''),
                              success=request.form.get('success') == 'on',
                              score=request.form.get('score', '0.8'), feedback=request.form.get('feedback', ''))
                if not secrets.compare_digest(request.form.get('nonce', ''), state.nonce):
                    error = 'This form has expired or has already been submitted. Check the result before trying again.'
                else:
                    state.nonce = secrets.token_urlsafe(24)
                    try:
                        if request.form.get('operation') == 'decide':
                            state.core.decide(values['situation'])
                        elif request.form.get('operation') == 'outcome':
                            try:
                                identifier = int(request.form.get('decision_id', ''))
                                score = float(values['score'].replace(',', '.'))
                            except ValueError:
                                raise ValueError('Enter a numeric score between 0 and 1.') from None
                            state.core.save_outcome(identifier, values['success'], score, values['feedback'])
                        else:
                            raise ValueError('Unknown operation.')
                        return redirect(url_for('index'), code=303)
                    except (ValueError, ModelError) as exc:
                        error = str(exc)
            snapshot = state.core.snapshot()
            latest = snapshot['decisions'][-1] if snapshot['decisions'] else None
            saved = latest and any(o['decision_id'] == latest['id'] for o in snapshot['outcomes'])
            if request.method == 'GET' and latest:
                values['situation'] = snapshot['situations'][-1]['text']
            ready = getattr(state.core.model, 'ready', False)
            testing = model is not None
            return render_template_string(PAGE, state=snapshot, latest=latest, saved=saved,
                                          error=error, values=values, nonce=state.nonce,
                                          ready=ready, testing=testing)

    @app.get('/favicon.ico')
    def favicon():
        return '', 204

    @app.errorhandler(413)
    def too_large(_):
        return 'Your input is too large. Go back and shorten the text.', 413

    return app


PAGE = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Recall — Decisions informed by experience</title>
<style>
:root{color-scheme:light;--ink:#182a32;--muted:#52656c;--line:#cfdbd7;--green:#176249;--bg:#f3f5ef}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}
main{max-width:1180px;margin:auto;padding:36px 28px 44px}header{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:30px}
.brand{font-size:27px;font-weight:780;letter-spacing:-1px}.brand span{color:var(--green)}.eyebrow{font-size:12px;font-weight:700;letter-spacing:1.8px;color:var(--muted);text-transform:uppercase}
h1{font-size:clamp(30px,4vw,48px);line-height:1.12;letter-spacing:-1.6px;margin:12px 0 16px;max-width:800px}p{margin:8px 0}.intro{max-width:750px;color:var(--muted)}
.pill{font-size:13px;border:1px solid var(--line);border-radius:24px;padding:7px 13px;background:white}
.flow{display:flex;flex-wrap:wrap;gap:10px 16px;margin:24px 0;color:var(--green);font-size:14px;font-weight:650}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:22px;align-items:start}
.card{background:#fff;border:1px solid var(--line);border-radius:18px;padding:24px;min-width:0}.card+.card{margin-top:20px}
h2{font-size:19px;margin:0 0 18px;letter-spacing:-.3px}h3{font-size:15px;margin:16px 0 8px}.step{color:var(--green);font:600 13px monospace;margin-right:10px}
label{display:block;font-size:14px;font-weight:650;margin-bottom:7px}textarea,input[type=text]{display:block;width:100%;border:1px solid #a6b8b1;border-radius:10px;padding:12px;font:inherit;color:var(--ink);background:#fff;resize:vertical}
textarea{min-height:100px}#situation{min-height:134px}#feedback{min-height:76px}.help{font-size:13px;color:var(--muted);margin:7px 0 15px}
button{border:0;border-radius:10px;background:var(--green);color:white;padding:12px 18px;font:650 15px system-ui;cursor:pointer;max-width:100%}button:disabled{background:#e7ece9;color:#52635b;cursor:default}
button:hover:not(:disabled){background:#114c38}:focus-visible{outline:3px solid #194fcd;outline-offset:3px}.row{display:grid;grid-template-columns:1fr 1fr;gap:16px;align-items:center;margin:16px 0}
.check{display:flex;gap:10px;align-items:center;margin:0}input[type=checkbox]{width:20px;height:20px;accent-color:var(--green)}
.empty{border:1px dashed var(--line);border-radius:10px;padding:20px;color:var(--muted);font-size:14px}.action{font-size:21px;font-weight:650;line-height:1.4;margin:14px 0;overflow-wrap:anywhere}
.tag{display:inline-block;background:#e7f3eb;color:#185139;padding:5px 9px;border-radius:6px;font-size:12px;font-weight:700}.reason,.wrap{overflow-wrap:anywhere}.feature{display:inline-block;background:#f0f3ef;border-radius:5px;margin:4px 5px 0 0;padding:3px 7px;font-size:12px}
.notice{padding:12px 16px;border:1px solid #cabfa8;border-radius:10px;background:#fff9eb;color:#60471f;font-size:14px;margin:16px 0}.error{background:#fff0ee;color:#8c2924;border-color:#e1a49d}
.memory{padding:12px 0;border-top:1px solid var(--line);font-size:14px;overflow-wrap:anywhere}.metrics{display:flex;gap:22px;margin-bottom:16px;font-size:13px;color:var(--muted)}.metrics strong{color:var(--ink);font-size:22px;display:block}
.evidence{background:#edf6f0;border-left:3px solid var(--green);padding:15px;border-radius:5px;font-size:14px;overflow-wrap:anywhere}footer{color:var(--muted);font-size:12px;margin-top:24px}
@media(max-width:740px){main{padding:22px 16px}.grid{grid-template-columns:1fr}header{align-items:flex-start;flex-wrap:wrap}.card{padding:20px}.row{grid-template-columns:1fr}h1{letter-spacing:-.8px}}

/* Compact workspace: keep all four steps visible on desktop. */
main{max-width:1800px;padding:16px 24px}header{justify-content:flex-start;gap:20px;margin-bottom:8px}
h1{font-size:18px;line-height:1.3;letter-spacing:0;margin:0}.brand{font-size:26px}
.flow{margin:8px 0 12px;font-size:13px;gap:8px 12px}
.grid{gap:14px}.card{padding:12px;border-radius:12px}.card+.card{margin-top:14px}
h2{font-size:17px;margin-bottom:8px}h3{font-size:14px;margin:12px 0 6px}
label{margin-bottom:4px}textarea,input[type=text]{padding:8px 10px;font-size:14px}
#situation{min-height:64px;height:72px}#feedback{min-height:36px;height:40px}
.help{font-size:12px;margin:4px 0 6px}.row{gap:12px;margin:6px 0}
button{padding:10px 16px;font-size:14px}.empty{padding:12px}.action{font-size:19px;margin:10px 0}
.notice{padding:8px 12px;margin:0 0 10px}.metrics{margin-bottom:8px}.memory{padding:8px 0}
.evidence{padding:12px}footer{font-size:11px;margin-top:10px;line-height:1.4}
@media(min-width:1100px) and (min-height:680px){
 main{height:100dvh;display:flex;flex-direction:column}
 .grid{flex:1;min-height:0;grid-template-columns:1fr 1fr 1.15fr;grid-template-rows:minmax(230px,0.9fr) minmax(285px,1.1fr);align-items:stretch}
 .grid>div{display:contents}.card{overflow:auto;scrollbar-gutter:stable}.card+.card{margin-top:0}
 .grid>div:first-child>.card:first-child{grid-column:1;grid-row:1}
 .grid>div:first-child>.card:last-child{grid-column:1;grid-row:2}
 #decision{grid-column:2;grid-row:1 / 3}#experience{grid-column:3;grid-row:1 / 3}
}
@media(max-width:740px){main{padding:12px}.grid{grid-template-columns:1fr}header{gap:6px 14px}h1{font-size:15px}.card{padding:14px}.row{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<header><div class="brand"><span>&#8627;</span> recall</div><h1>Decisions informed by experience</h1></header>
<div class="flow"><span>01 · Situation</span><span>→</span><span>02 · Decision</span><span>→</span><span>03 · Outcome</span><span>→</span><span>04 · Experience transfer</span></div>
{% if testing %}<div class="notice" id="mode">Test double — no requests are sent to Nebius.</div>
{% elif not ready %}<div class="notice" id="mode">The model is not connected yet. Configure Nebius access to get your first decision.</div>{% endif %}
{% if error %}<div class="notice error" id="error" role="alert">{{ error }}</div>{% endif %}
<div class="grid"><div>
<section class="card"><h2><span class="step">01</span>New situation</h2>
<form method="post" id="decision-form"><input type="hidden" name="nonce" value="{{ nonce }}"><input type="hidden" name="operation" value="decide">
<label for="situation">Situation</label><textarea id="situation" name="situation" maxlength="2000" required aria-describedby="situation-help">{{ values.situation }}</textarea>
<p class="help" id="situation-help">Describe the problem and any relevant constraints. Up to 2,000 characters.</p>
<button id="decide" type="submit">Get decision</button></form></section>
<section class="card" aria-labelledby="outcome-title"><h2 id="outcome-title"><span class="step">03</span>Actual outcome</h2>
<p class="help">{% if latest %}For decision #{{ latest.id }}. {% if saved %}Outcome saved.{% else %}Rate the outcome after trying the action.{% endif %}{% else %}Get a decision first.{% endif %}</p>
<form method="post" id="outcome-form"><input type="hidden" name="nonce" value="{{ nonce }}"><input type="hidden" name="operation" value="outcome"><input type="hidden" name="decision_id" value="{{ latest.id if latest else '' }}">
<div class="row"><label class="check" for="success"><input id="success" type="checkbox" name="success" {% if values.success %}checked{% endif %}>Success</label>
<div><label for="score">Score · 0–1</label><input id="score" name="score" type="text" inputmode="decimal" maxlength="20" value="{{ values.score }}" required></div></div>
<label for="feedback">Feedback</label><textarea id="feedback" name="feedback" maxlength="1000">{{ values.feedback }}</textarea>
<p class="help">An experience is eligible for transfer when the outcome is successful and the score is at least 0.5.</p>
<button id="save" type="submit" {% if not latest or saved %}disabled{% endif %}>Save outcome</button></form></section>
</div><div>
<section class="card" id="decision" aria-labelledby="decision-title"><h2 id="decision-title"><span class="step">02</span>Decision</h2>
{% if latest %}<span class="tag" id="source">{% if latest.source == 'MODEL' %}MODEL{% else %}EXPERIENCE{% endif %}</span>
<div class="action" id="action">{{ latest.action }}</div><p class="reason" id="reason">{{ latest.reason }}</p>
<h3>Situation features</h3><div class="wrap" id="features">{% for f in state.situations[-1].features %}<span class="feature">{{ f }}</span>{% endfor %}</div>
{% else %}<div class="empty">The suggested action and its rationale will appear here.</div>{% endif %}</section>
<section class="card" id="experience" aria-labelledby="experience-title"><h2 id="experience-title"><span class="step">04</span>Experience</h2>
<div class="metrics"><div><strong id="decision-count">{{ state.decisions|length }}</strong>decisions</div><div><strong id="experience-count">{{ state.experiences|length }}</strong>saved experiences</div></div>
{% for exp in state.experiences|reverse %}<div class="memory"><strong>Experience #{{ exp.id }} · decision #{{ exp.decision_id }}</strong><p>{{ exp.action }}</p><span>{% if exp.success %}Successful{% else %}Unsuccessful{% endif %} · score {{ '%g'|format(exp.score) }}</span></div>
{% else %}<div class="empty">Save an outcome to create your first experience.</div>{% endfor %}
<h3>Transfer evidence</h3>
<div id="evidence">{% if latest and latest.evidence %}{% set ev = latest.evidence %}<div class="evidence">
{% if ev.status %}
<strong id="transfer-status">{{ ev.status }}</strong><p>{{ ev.reason }}</p>
<p>Experience not transferred: #{{ ev.experience.id }}. Similarity: <span id="similarity">{{ '%.2f'|format(ev.similarity) }}</span>.</p>
<p>The previous action was not transferred. The model provided a new decision.</p>
{% else %}
<strong>Source: experience #{{ ev.experience.id }}</strong><p>Situation #{{ ev.experience.situation_id }} → decision #{{ ev.experience.decision_id }}</p>
<p>Similarity: <span id="similarity">{{ '%.2f'|format(ev.similarity) }}</span> · score: {{ '%g'|format(ev.score) }}</p>
<p>Source features: {{ ev.experience.situation_features|join(', ') }}</p><p>Selected action: {{ ev.action }}</p>
<p>No additional model request was needed.</p>{% endif %}</div>{% else %}<p class="help">Transfer evidence will appear when a new situation reuses a relevant successful experience.</p>{% endif %}</div></section>
</div></div>
<footer>Memory persists when you refresh the page. It is cleared when the server restarts or after 24 hours of inactivity. Similarity is based on words; review differences in meaning yourself. Recall suggests an action. You decide whether to take it.</footer>
</main><script>
document.querySelectorAll('form').forEach(form=>form.addEventListener('submit',()=>{
form.querySelector('button').disabled=true;
form.querySelector('button').textContent='Processing…';
}));
</script></body></html>'''

if __name__ == '__main__':
    create_app().run(host='127.0.0.1', port=5000, debug=False)
