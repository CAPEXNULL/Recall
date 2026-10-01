"""Acceptance scenarios. Default: local checks only, never a live model call.

Full run: RECALL_FULL_RUN=1, RECALL_MODEL_LOCK pointing to verified model lock.
Test dependencies: pip install playwright==1.52.0 pathspec==0.12.1;
playwright install chromium. Artifacts must stay outside the public source tree.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from threading import Thread
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app import SCENARIO, create_app
from model_client import NebiusClient, ModelError, OUTPUT_SCHEMA, validate_output
from recall_core import Recall, critical_mismatch, features_for, similarity

FULL = os.environ.get('RECALL_FULL_RUN') == '1'
RUN_ID = os.environ.get('RECALL_RUN_ID', 'local-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S'))
EVIDENCE = Path(os.environ.get('RECALL_EVIDENCE_DIR', str(ROOT.parent / 'evidence' / RUN_ID))).resolve()
TRACE = Path(os.environ.get('RECALL_TRACE_PATH', str(EVIDENCE / 'TRACE.jsonl'))).resolve()
if EVIDENCE.is_relative_to(ROOT) or TRACE.is_relative_to(ROOT):
    raise SystemExit('Evidence and TRACE must be outside the public recall directory.')


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


@contextmanager
def observe(test_id, operation, criterion):
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc).isoformat()
    detail = {}
    status, failure = 'PASS', None
    try:
        yield detail
    except Exception as exc:
        status, failure = 'FAIL', type(exc).__name__
        detail['failure_type'] = failure
        raise
    finally:
        path = EVIDENCE / f'{test_id}-{operation}.json'
        path.write_text(json.dumps(detail, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        trace = TRACE
        trace.parent.mkdir(parents=True, exist_ok=True)
        rows = trace.read_text(encoding='utf-8').splitlines() if trace.exists() else []
        seq = len(rows) + 1
        state = {'applicable': False, 'reason': 'Isolated test fixture; see evidence for state transitions'}
        row = dict(run_id=RUN_ID, sequence=seq, operation_id=f'{RUN_ID}-{seq:04}',
                   started_at=start, finished_at=datetime.now(timezone.utc).isoformat(),
                   component='TEST:' + test_id, action=operation, target_ref=test_id,
                   input={'fixture':'demo/scenario.json','mode':'full' if FULL else 'local'},
                   source_ref=['tests/test_recall.py'], state_before=state,
                   pass_criterion=criterion, result={'test_id':test_id,'status':status}, state_after=state,
                   evidence={'path':str(path.resolve()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
                   status=status)
        if failure:
            row.update(defect_id=test_id + '-' + operation, defect_state='OPEN', failure_reason=failure)
        with trace.open('a', encoding='utf-8') as f:
            f.write(canonical(row) + '\n')


class FakeModel:
    ready = True

    def __init__(self):
        self.calls = 0

    def decide(self, text, features):
        self.calls += 1
        return dict(features=features, action='Вернуться к предыдущей стабильной версии сервиса.',
                    reason='Проблема появилась после обновления; проверьте восстановление задержки после отката.')


class TransferObjectRegression(unittest.TestCase):
    def transfer(self, old, new, action):
        original = f'После обновления сервис {old} отвечает медленно, очередь запросов растёт. Доступен откат версии.'
        current = f'После обновления сервис {new} отвечает медленно, очередь запросов растёт. Доступен откат версии.'
        model = FakeModel()
        with patch.object(model, 'decide', return_value=dict(
                features=features_for(original), action=action, reason='Проверенное действие.')) as call:
            core = Recall(model)
            first = core.decide(original)
            experience = core.save_outcome(first['id'], True, .8, 'Демонстрационный исход')
            result = core.decide(current)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(result['source'], 'EXPERIENCE')
        self.assertEqual(result['evidence']['experience'], experience)
        self.assertEqual(result['evidence']['similarity'],
                         similarity(features_for(original), features_for(current)))
        self.assertEqual(result['evidence']['action'], result['action'])
        self.assertEqual(core.snapshot()['experiences'][0], experience)
        self.assertEqual(core.snapshot()['decisions'][0], first)
        return result['action']

    def test_named_service_transfer(self):
        for old, new in [('Альфа', 'Бета'), ('Орион', 'Вега'),
                         ('Payments-v2', 'Billing-v3'), ('«Север API»', '«Юг API»')]:
            with self.subTest(old=old, new=new):
                self.assertEqual(self.transfer(old, new,
                    f'Откатить версию сервиса {old} до предыдущей'),
                    f'Откатить версию сервиса {new} до предыдущей')

    def test_ambiguous_object_keeps_action(self):
        action = 'Откатить версию сервиса Альфа до предыдущей'
        for new in ['Бета и Гамма', 'Бета или сервис Гамма', 'без имени', 'Бета Север']:
            with self.subTest(new=new):
                self.assertEqual(self.transfer('Альфа', new, action), action)
        self.assertEqual(self.transfer('Альфа и Гамма', 'Бета и Гамма', action), action)

    def test_other_action_text_preserved(self):
        action = 'Откатить версию сервиса Альфа до предыдущей; сохранить журнал Альфа.'
        self.assertEqual(self.transfer('Альфа', 'Бета', action),
                         'Откатить версию сервиса Бета до предыдущей; сохранить журнал Альфа.')
        generic = 'Вернуться к предыдущей стабильной версии сервиса.'
        self.assertEqual(self.transfer('Альфа', 'Бета', generic), generic)
        ambiguous = 'Откатить версию сервиса Альфа Север до предыдущей'
        self.assertEqual(self.transfer('Альфа', 'Бета', ambiguous), ambiguous)


class CriticalMismatchRegression(unittest.TestCase):
    original = 'После обновления сервис Орион отвечает медленно, очередь запросов растёт. Доступен откат версии.'
    current = original.replace('Орион', 'Вега').replace('Доступен откат версии', 'Откат версии недоступен')
    action = 'Откатить версию сервиса Орион до предыдущей'
    fallback = 'Проверить журнал ошибок сервиса Вега'
    status = 'TRANSFER BLOCKED: critical mismatch'

    def model(self):
        owner = self
        class ScriptedModel:
            ready = True
            calls = 0
            def decide(self, text, features):
                self.calls += 1
                return dict(features=features, action=owner.action if self.calls == 1 else owner.fallback,
                            reason='Локальный тестовый ответ.')
        return ScriptedModel()

    def prepared(self):
        model = self.model()
        core = Recall(model)
        first = core.decide(self.original)
        exp = core.save_outcome(first['id'], True, .8, 'Тестовый исход')
        return core, model, exp

    def test_blocked_fallback_and_determinism(self):
        snapshots = []
        for _ in range(2):
            core, model, exp = self.prepared()
            result = core.decide(self.current)
            self.assertEqual(result['source'], 'MODEL')
            self.assertEqual(result['action'], self.fallback)
            self.assertEqual(model.calls, 2)  # Initial decision + ordinary fallback only.
            ev = result['evidence']
            self.assertEqual(ev['status'], self.status)
            self.assertIn('недоступен', ev['reason'])
            self.assertNotIn('action', ev)
            self.assertEqual(ev['experience'], exp)
            self.assertGreaterEqual(ev['similarity'], .5)
            self.assertEqual(ev['similarity'], similarity(features_for(self.original), features_for(self.current)))
            self.assertEqual(core.snapshot()['experiences'], [exp])
            snapshots.append(core.snapshot())
        self.assertEqual(*snapshots)

    def test_constraint_scope(self):
        for text in ['Откат недоступен.', 'Недоступен откат версии.',
                     'Откат версии не доступен.', 'Нельзя откатить версию.',
                     'Откат запрещён.']:
            with self.subTest(text=text):
                self.assertIsNotNone(critical_mismatch(self.action, self.original, text))
        for text in ['Доступен откат версии.', 'Журнал недоступен.',
                     'Откат не запрещён.', 'Откат не невозможен.']:
            with self.subTest(text=text):
                self.assertIsNone(critical_mismatch(self.action, self.original, text))
        self.assertIsNotNone(critical_mismatch('Выполнить удаление файла',
                                              'Удаление разрешено.', 'Удаление запрещено.'))
        self.assertIsNone(critical_mismatch(self.action, 'Откат запрещён.', 'Откат запрещён.'))
        self.assertIsNone(critical_mismatch('Выполнить откат версии', '', 'Нельзя выполнить удаление.'))
        self.assertIsNotNone(critical_mismatch('Выполнить откат версии', '', 'Нельзя выполнить откат версии.'))
        self.assertIsNotNone(critical_mismatch('Выполнить откат версии', '', 'Запрещено выполнить откат версии.'))

    def test_fallback_error_preserves_state_and_status(self):
        core, model, _ = self.prepared()
        before = core.snapshot()
        with patch.object(model, 'decide', side_effect=ModelError('Тестовая ошибка')) as call:
            with self.assertRaisesRegex(ModelError, self.status):
                core.decide(self.current)
            self.assertEqual(call.call_count, 1)
        self.assertEqual(core.snapshot(), before)

    def test_demo_flow_valid_then_blocked(self):
        model = self.model()
        app = create_app(model)
        client = app.test_client()
        self.assertEqual(client.get('/').status_code, 200)
        state = next(iter(app.extensions['recall_states'].values()))
        def post(**data):
            response = client.post('/', data=dict(nonce=state.nonce, **data), follow_redirects=True)
            self.assertEqual(response.status_code, 200)
            return response.get_data(as_text=True)
        post(operation='decide', situation=self.original)
        post(operation='outcome', decision_id='1', success='on', score='.8', feedback='Тестовый исход')
        html = post(operation='decide', situation=self.original.replace('Орион', 'Вега'))
        self.assertIn('Откатить версию сервиса Вега до предыдущей', html)
        self.assertIn('No additional model request was needed.', html)
        self.assertNotIn(self.status, html)
        self.assertEqual(model.calls, 1)
        html = post(operation='decide', situation=self.current)
        self.assertIn(self.status, html)
        self.assertIn(self.fallback, html)
        self.assertNotIn('Selected action:', html)
        self.assertNotIn('No additional model request was needed.', html)
        self.assertEqual(model.calls, 2)
        self.assertIn(self.status, client.get('/').get_data(as_text=True))
        post(operation='outcome', decision_id='3', success='on', score='.8', feedback='Новый исход')
        self.assertEqual(state.core.snapshot()['experiences'][-1]['action'], self.fallback)


class Acceptance(unittest.TestCase):
    def test_01_core_outcome(self):
        model = FakeModel()
        core = Recall(model)
        with observe('TEST-01', 'decision', 'Empty state accepts A, MODEL decision is linked to saved situation; exactly one model call') as ev:
            ev['before'] = core.snapshot()
            self.assertEqual(ev['before']['experiences'], [])
            decision = core.decide(SCENARIO['situation_a'])
            state = core.snapshot()
            self.assertEqual(decision['source'], 'MODEL')
            self.assertEqual(decision['situation_id'], state['situations'][0]['id'])
            self.assertEqual(state['situations'][0]['features'], features_for(SCENARIO['situation_a']))
            self.assertEqual(model.calls, 1)
            ev.update(after=state, model_calls=model.calls)
        with observe('TEST-01', 'outcome', 'One outcome creates one experience linked to original decision and features, retaining score and action') as ev:
            ev['before'] = core.snapshot()
            exp = core.save_outcome(decision['id'], **SCENARIO['outcome'])
            after = core.snapshot()
            self.assertEqual(len(after['outcomes']), 1)
            self.assertEqual(exp['action'], decision['action'])
            self.assertEqual(exp['decision_id'], decision['id'])
            self.assertEqual(exp['situation_features'], after['situations'][0]['features'])
            self.assertEqual(exp['score'], 0.8)
            ev['after'] = after
        with observe('TEST-01', 'validation', 'Invalid input, duplicate outcome or invalid model response cannot mutate state; score boundaries accepted; snapshots isolated') as ev:
            before = core.snapshot()
            cases = [('',), ('!!!',), ('a' * 2001,)]
            for args in cases:
                with self.assertRaises(ValueError):
                    core.decide(*args)
            bad_outcomes = [
                (1, True, 0.8, ''), (0, True, 0.8, ''), (True, True, 0.8, ''),
                (1, 'yes', 0.8, ''), (1, True, math.nan, ''),
                (1, True, math.inf, ''), (1, True, -0.1, ''),
                (1, True, 1.1, ''), (1, True, True, ''), (1, True, .8, 'x' * 1001)]
            for args in bad_outcomes:
                with self.assertRaises(ValueError):
                    core.save_outcome(*args)
                self.assertEqual(core.snapshot(), before)
            snapshot = core.snapshot()
            snapshot['experiences'][0]['action'] = 'mutated'
            self.assertEqual(core.snapshot(), before)
            boundary = Recall(FakeModel())
            a = boundary.decide('alpha')
            b = boundary.decide('beta')
            boundary.save_outcome(a['id'], False, 0, '')
            boundary.save_outcome(b['id'], True, 1, 'x' * 1000)
            self.assertEqual([o['score'] for o in boundary.snapshot()['outcomes']], [0.0, 1.0])
            with patch.object(model, 'decide', return_value={'features': ['wrong'], 'action': 'a', 'reason': 'r'}):
                with self.assertRaises(ModelError):
                    core.decide('unrelated')
            self.assertEqual(core.snapshot(), before)
            ev.update(rejected_situations=len(cases), rejected_outcomes=len(bad_outcomes),
                      state_unchanged=True, score_boundaries=[0,1], snapshot_isolated=True)
        with observe('TEST-01', 'adapter', 'Adapter sends exact configured Nemotron model and JSON schema; accepts valid response and rejects malformed/refused/truncated content, with no network') as ev:
            features = features_for('Проверка сервиса')
            valid = dict(features=features, action='Проверить журнал.', reason='Нужна причина ошибки.')
            with patch.dict(os.environ, {'NEBIUS_API_KEY':'test-only-placeholder', 'NEBIUS_MODEL_ID':'nvidia/test-nemotron'}):
                client = NebiusClient()
                reply = {'choices':[{'finish_reason':'stop','message':{'content':canonical(valid)}}]}
                with patch.object(client, '_request', return_value=(reply, 200)) as transport:
                    self.assertEqual(client.decide('Проверка сервиса', features), valid)
                    payload = transport.call_args.args[1]
                    self.assertEqual(payload['model'], 'nvidia/test-nemotron')
                    self.assertEqual(payload['response_format']['json_schema']['schema'], OUTPUT_SCHEMA)
                for response in [
                    {}, {'choices':[]},
                    {'choices':[{'finish_reason':'length','message':{'content':canonical(valid)}}]},
                    {'choices':[{'finish_reason':'stop','message':{'refusal':'no','content':None}}]},
                    {'choices':[{'finish_reason':'stop','message':{'content':'not-json'}}]},
                ]:
                    with patch.object(client, '_request', return_value=(response, 200)):
                        with self.assertRaises(ModelError):
                            client.decide('Проверка сервиса', features)
                for bad in [[], {**valid,'extra':1}, {**valid,'features':[]}, {**valid,'action':''}, {**valid,'reason':None}]:
                    with self.assertRaises(ModelError):
                        validate_output(bad, features)
            ev.update(schema=OUTPUT_SCHEMA, valid_response=valid, invalid_responses_rejected=10, network_calls=0)

    def test_02_transfer_repeatability(self):
        with observe('TEST-02', 'transfer', 'New exact text B reuses A action with similarity >=0.5, full provenance, and only one model call total') as ev:
            model = FakeModel()
            core = Recall(model)
            first = core.decide(SCENARIO['situation_a'])
            exp = core.save_outcome(first['id'], **SCENARIO['outcome'])
            self.assertNotIn(SCENARIO['situation_b'], [s['text'] for s in core.snapshot()['situations']])
            decision = core.decide(SCENARIO['situation_b'])
            self.assertEqual(decision['source'], 'EXPERIENCE')
            self.assertEqual(decision['action'], first['action'])
            self.assertGreaterEqual(decision['evidence']['similarity'], .5)
            self.assertEqual(decision['evidence']['experience'], exp)
            self.assertEqual(model.calls, 1)
            reference = canonical(core.snapshot())
            ev.update(state=core.snapshot(), model_calls=model.calls)
        with observe('TEST-02', 'repeatability', 'Fresh state, same inputs and test double yield byte-identical canonical JSON') as ev:
            repeat = Recall(FakeModel())
            a = repeat.decide(SCENARIO['situation_a'])
            repeat.save_outcome(a['id'], **SCENARIO['outcome'])
            repeat.decide(SCENARIO['situation_b'])
            result = canonical(repeat.snapshot())
            self.assertEqual(result, reference)
            ev.update(canonical_sha256=hashlib.sha256(result.encode('utf-8')).hexdigest(), identical=True)
        with observe('TEST-02', 'selection', 'Jaccard boundary 0.5 is inclusive; failed/low-score outcomes excluded; rank by similarity, score, NFC action, integer ID') as ev:
            self.assertEqual(features_for('CAFÉ cafe\u0301'), ['café'])
            self.assertEqual(similarity([], []), 0)
            self.assertEqual(similarity(['a'], ['a','b']), .5)
            # Controlled persisted-memory fixtures isolate each ranking dimension.
            def select(records, query='a b'):
                core = Recall(FakeModel())
                core._state['experiences'] = [dict(id=i, situation_id=i, decision_id=i,
                    situation_features=f, action=a, success=s, score=v) for i,f,a,s,v in records]
                return core.decide(query)
            cases = [
                ([(1,['a'],'A',True,.5)], 1),
                ([(1,['a'],'A',False,1)], None),
                ([(1,['a'],'A',True,.49)], None),
                ([(1,['c'],'A',True,1)], None),
                ([(1,['a'],'A',True,1),(2,['a','b'],'Z',True,.5)], 2),
                ([(1,['a'],'A',True,.5),(2,['b'],'Z',True,.9)], 2),
                ([(1,['a'],'Z',True,.8),(2,['b'],'A',True,.8)], 2),
                ([(9,['a'],'A',True,.8),(2,['b'],'A',True,.8)], 2),
            ]
            outcomes = []
            for records, expected in cases:
                result = select(records)
                actual = result['evidence']['experience']['id'] if result['evidence'] else None
                self.assertEqual(actual, expected)
                outcomes.append(dict(expected_id=expected, actual_id=actual, source=result['source']))
            ev['cases'] = outcomes

    @unittest.skipUnless(FULL, 'Local mode: live integration remains BLOCKED, never represented as PASS')
    def test_03_live(self):
        with observe('TEST-03', 'live', 'Locked listed Nemotron responds once, validates schema and is used in stored MODEL decision; metadata contains no secret') as ev:
            lock = json.loads(Path(os.environ['RECALL_MODEL_LOCK']).read_text(encoding='utf-8'))
            client = NebiusClient()
            self.assertTrue(client.ready)
            self.assertEqual(client.model_id, lock['model_id'])
            self.assertIn(client.model_id, lock['available_models'])
            self.assertTrue(lock['json_mode_supported'])
            core = Recall(client)
            result = core.decide(SCENARIO['situation_a'])
            self.assertEqual(result['source'], 'MODEL')
            self.assertEqual(core.snapshot()['decisions'][0], result)
            self.assertEqual(client.last_metadata['http_status'], 200)
            self.assertTrue(client.last_metadata['schema_valid'])
            ev.update(api=client.last_metadata, state=core.snapshot())
            self.assertNotIn(os.environ['NEBIUS_API_KEY'], canonical(ev))

    def test_04_ui_repository(self):
        from playwright.sync_api import sync_playwright
        from werkzeug.serving import make_server, WSGIRequestHandler

        class QuietHandler(WSGIRequestHandler):
            def log(self, *args, **kwargs):
                pass

        app = create_app(FakeModel())
        server = make_server('127.0.0.1', 0, app, request_handler=QuietHandler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        url = f'http://127.0.0.1:{server.server_port}/'
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                context = browser.new_context(viewport={'width':1440,'height':1000})
                page = context.new_page()
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                page.on('console', lambda msg: errors.append(msg.text) if msg.type == 'error' else None)
                with observe('TEST-04', 'controls', 'Exactly six mapped visible inputs/buttons; labels; initial data; keyboard focus and native empty-input validation') as ev:
                    page.goto(url)
                    self.assertIn('Test double', page.locator('#mode').inner_text())
                    self.assertEqual(page.locator('#situation').input_value(), SCENARIO['situation_a'])
                    self.assertFalse(page.locator('#success').is_checked())
                    self.assertEqual(page.locator('#score').input_value(), '0.8')
                    self.assertTrue(page.locator('#save').is_disabled())
                    controls = page.locator('input:not([type=hidden]),textarea,button').evaluate_all('(els)=>els.map(e=>({id:e.id,type:e.type,label:e.labels?.[0]?.innerText||e.innerText,disabled:e.disabled}))')
                    self.assertEqual({c['id'] for c in controls}, {'situation','decide','success','score','feedback','save'})
                    self.assertTrue(all(c['label'] for c in controls))
                    page.keyboard.press('Tab')
                    self.assertEqual(page.locator(':focus').get_attribute('id'), 'situation')
                    self.assertEqual(page.locator(':focus').evaluate('e=>getComputedStyle(e).outlineWidth'), '3px')
                    reached = ['situation']
                    for _ in range(4):
                        page.keyboard.press('Tab')
                        reached.append(page.locator(':focus').get_attribute('id'))
                    self.assertEqual(reached, ['situation','decide','success','score','feedback'])
                    page.locator('#situation').fill('')
                    page.locator('#decide').click()
                    self.assertFalse(page.locator('#situation').evaluate('e=>e.checkValidity()'))
                    self.assertEqual(page.locator('#decision-count').inner_text(), '0')
                    ev.update(controls=controls, keyboard_order=reached, empty_input_rejected=True)
                with observe('TEST-04', 'demo', 'A -> MODEL -> successful outcome -> new B -> EXPERIENCE; UI renders actual core action, provenance and similarity; reload retains memory') as ev:
                    page.locator('#situation').fill(SCENARIO['situation_a'])
                    page.locator('#decide').click()
                    page.wait_for_selector('#source')
                    self.assertIn('MODEL', page.locator('#source').inner_text())
                    action = page.locator('#action').inner_text()
                    # Before saving, reject a malformed score and preserve decision.
                    page.locator('#score').fill('NaN')
                    page.locator('#save').click()
                    page.wait_for_selector('#error')
                    self.assertIn('between 0 and 1', page.locator('#error').inner_text())
                    self.assertEqual(page.locator('#action').inner_text(), action)
                    self.assertEqual(page.locator('#experience-count').inner_text(), '0')
                    page.locator('#success').check()
                    page.locator('#score').fill('0,8')
                    page.locator('#feedback').fill(SCENARIO['outcome']['feedback'])
                    page.locator('#save').click()
                    page.wait_for_function("document.querySelector('#experience-count').textContent === '1'")
                    self.assertTrue(page.locator('#save').is_disabled())
                    page.locator('#situation').fill(SCENARIO['situation_b'])
                    page.locator('#decide').click()
                    page.wait_for_selector('#similarity')
                    self.assertIn('EXPERIENCE', page.locator('#source').inner_text())
                    self.assertEqual(page.locator('#action').inner_text(), action)
                    self.assertGreaterEqual(float(page.locator('#similarity').inner_text()), .5)
                    session_state = next(iter(app.extensions['recall_states'].values()))
                    saved_state = session_state.core.snapshot()
                    self.assertEqual(saved_state['decisions'][-1]['action'], action)
                    self.assertEqual(session_state.core.model.calls, 1)
                    self.assertEqual(saved_state['outcomes'][0]['feedback'], SCENARIO['outcome']['feedback'])
                    page.reload()
                    self.assertEqual(page.locator('#decision-count').inner_text(), '2')
                    self.assertEqual(page.locator('#experience-count').inner_text(), '1')
                    ev.update(state=saved_state, displayed_evidence=page.locator('#evidence').inner_text(),
                              malformed_score_preserved_result=True, reload_retained=True)
                with observe('TEST-04', 'validation-replay-isolation', 'Invalid situation preserves last result; same nonce cannot create twice; separate browser session starts empty') as ev:
                    page.locator('#situation').fill('!!!')
                    page.locator('#decide').click()
                    page.wait_for_selector('#error')
                    self.assertIn('between 1 and 200', page.locator('#error').inner_text())
                    self.assertEqual(page.locator('#action').inner_text(), action)
                    self.assertFalse(page.locator('#decide').is_disabled())
                    self.assertEqual(page.locator('#decision-count').inner_text(), '2')
                    data = page.locator('#decision-form').evaluate('(e)=>Object.fromEntries(new FormData(e))')
                    data['situation'] = SCENARIO['situation_b']
                    context.request.post(url, form=data)
                    replay = context.request.post(url, form=data)
                    self.assertIn('already been submitted', replay.text())
                    self.assertEqual(len(session_state.core.snapshot()['decisions']), 3)
                    isolated = browser.new_context()
                    other = isolated.new_page()
                    other.goto(url)
                    self.assertEqual(other.locator('#decision-count').inner_text(), '0')
                    isolated.close()
                    page.reload()
                    ev.update(invalid_text_preserved=True, replay_rejected=True, independent_session_empty=True)
                with observe('TEST-04', 'layout-accessibility', 'Desktop/mobile/200% viewport: no clipping, overlap or overflow; labelled controls; text contrast >=4.5; screenshots; console/runtime errors zero') as ev:
                    layouts = []
                    for name, width, height in [('desktop',1440,1000),('mobile',390,844),('zoom200',720,500)]:
                        page.set_viewport_size({'width':width,'height':height})
                        measured = page.evaluate('''()=>{
                          const els=[...document.querySelectorAll('textarea,input:not([type=hidden]),button,.card')];
                          const rects=els.map(e=>{const r=e.getBoundingClientRect();return {id:e.id||e.tagName,x:r.x,y:r.y,w:r.width,h:r.height}});
                          const controls=rects.filter(r=>r.id!=='SECTION');
                          let overlap=false;
                          for(let i=0;i<controls.length;i++)for(let j=i+1;j<controls.length;j++){
                            const a=controls[i],b=controls[j];
                            if(Math.min(a.x+a.w,b.x+b.w)>Math.max(a.x,b.x)+1 && Math.min(a.y+a.h,b.y+b.h)>Math.max(a.y,b.y)+1)overlap=true;
                          }
                          return {width:innerWidth,scrollWidth:document.documentElement.scrollWidth,rects,overlap};
                        }''')
                        self.assertLessEqual(measured['scrollWidth'], width)
                        self.assertFalse(measured['overlap'])
                        self.assertTrue(all(r['x']>=-1 and r['x']+r['w']<=width+1 and r['w']>0 and r['h']>0 for r in measured['rects']))
                        shot = EVIDENCE / f'ui-{name}.png'
                        page.screenshot(path=str(shot), full_page=True)
                        measured['screenshot'] = dict(path=str(shot.resolve()),sha256=hashlib.sha256(shot.read_bytes()).hexdigest())
                        layouts.append(measured)
                    # These fixed rendered design tokens cover body, secondary text,
                    # primary action and the transfer card; disabled controls are exempt.
                    def luminance(hex_value):
                        channels = [int(hex_value[i:i+2],16)/255 for i in (0,2,4)]
                        return sum(w*(c/12.92 if c<=.04045 else ((c+.055)/1.055)**2.4) for w,c in zip([.2126,.7152,.0722],channels))
                    pairs=[('182a32','ffffff'),('52656c','ffffff'),('52656c','f3f5ef'),('ffffff','176249'),('185139','e7f3eb'),('182a32','edf6f0')]
                    ratios=[]
                    for fg,bg in pairs:
                        a,b=sorted([luminance(fg),luminance(bg)])
                        ratio=(b+.05)/(a+.05)
                        self.assertGreaterEqual(ratio,4.5)
                        ratios.append(dict(fg=fg,bg=bg,ratio=ratio))
                    self.assertEqual(errors, [])
                    ev.update(layouts=layouts, contrast=ratios, console_runtime_errors=errors)
                browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()
        with observe('TEST-04', 'repository', 'Audit all public files, external evidence paths, required README content and license state; gitignore semantics exclude secrets and work artifacts without excluding source') as ev:
            from pathspec import GitIgnoreSpec
            required=['README.md','LICENSE_SELECTION_REQUIRED.md','.gitignore','.env.example','requirements.txt','app.py','recall_core.py','model_client.py','demo/scenario.json','tests/test_recall.py']
            manifest={}
            for name in required:
                data=(ROOT/name).read_bytes()
                self.assertTrue(data)
                manifest[name]=hashlib.sha256(data).hexdigest()
            template=(ROOT/'.env.example').read_text()
            self.assertTrue(all(line.endswith('=') and line.count('=')==1 for line in template.splitlines()))
            self.assertFalse(EVIDENCE.is_relative_to(ROOT))
            self.assertFalse(TRACE.is_relative_to(ROOT))
            # Only generated bytecode and the documented virtual environment are
            # outside the deliverable allowlist. Hidden/unlisted files must fail.
            actual=[]
            for path in ROOT.rglob('*'):
                relative=path.relative_to(ROOT)
                if any(part in {'.venv','__pycache__'} for part in relative.parts):
                    continue
                self.assertFalse(path.is_symlink(), str(relative))
                if path.is_file():
                    actual.append(relative.as_posix())
            self.assertEqual(set(actual),set(required))
            ignore=GitIgnoreSpec.from_lines((ROOT/'.gitignore').read_text().splitlines())
            excluded=['TRACE.jsonl','evidence/check.json','test-results/ui.png',
                      '.env','.env.local','nested/.env','nested/TRACE.jsonl',
                      'nested/evidence/check.json','session.log']
            matching={name:ignore.match_file(name) for name in excluded+required}
            self.assertTrue(all(matching[name] for name in excluded))
            self.assertFalse(any(matching[name] for name in required))
            content='\n'.join((ROOT/name).read_text(encoding='utf-8') for name in required)
            for private in ['-----BEGIN ' + 'PRIVATE KEY-----','-----BEGIN ' + 'RSA PRIVATE KEY-----']:
                self.assertNotIn(private,content)
            key=os.environ.get('NEBIUS_API_KEY')
            if key:
                self.assertNotIn(key, content)
            readme=(ROOT/'README.md').read_text(encoding='utf-8')
            self.assertIn('python app.py',readme)
            self.assertIn('SUBMISSION_READY = false',readme)
            claims=['new, independent entry','Nebius Token Factory','NVIDIA Nemotron',
                    '## Technology feedback','**Used / purpose:**','**Worked well locally:**',
                    '**Difficulty:**','**Onboarding:**','**Suitability:**']
            self.assertTrue(all(claim in readme for claim in claims))
            ev.update(manifest=manifest, actual_public_files=sorted(actual),
                      environment_values_empty=True, private_key_markers_absent=True,
                      known_environment_key_absent=True, license_selection_pending=True,
                      readme_required_claims=claims, ignore_matching=matching,
                      artifacts_outside_public_tree=True)


if __name__ == '__main__':
    if FULL and not (os.environ.get('NEBIUS_API_KEY') and os.environ.get('NEBIUS_MODEL_ID') and os.environ.get('RECALL_MODEL_LOCK')):
        raise SystemExit('BLOCKED: credentials and verified model lock are required before a full run.')
    unittest.main(verbosity=2, failfast=True)
