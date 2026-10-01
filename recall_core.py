"""Small, deterministic outcome-backed memory. No model weights are changed."""
from copy import deepcopy
import math
import re
import unicodedata

TRANSFER_THRESHOLD = 0.50
SUCCESS_SCORE_THRESHOLD = 0.50


def normalize(value):
    return unicodedata.normalize('NFC', value.strip())


def features_for(text):
    text = unicodedata.normalize('NFC', normalize(text).casefold())
    return sorted(set(re.findall(r'[^\W_]+', text, flags=re.UNICODE)))


def similarity(left, right):
    a, b = set(left), set(right)
    return len(a & b) / len(a | b) if a | b else 0.0


def adapt_transferred_action(action, original, current):
    """Bind an explicit service name only when the surrounding situation agrees.

    Free-form or ambiguous object descriptions keep the confirmed action intact.
    Only service-qualified names in the action are changed, never bare mentions.
    """
    service = re.compile(
        r'(?<!\w)(?i:сервис(?:а|у|ом|е)?|services?)\s+'
        r'(?P<name>«[^«»\n]+»|"[^"\n]+"|[A-ZА-ЯЁ][\w-]*)(?![\w-])')
    before, after = list(service.finditer(original)), list(service.finditer(current))
    if len(before) != 1 or len(after) != 1:
        return action
    old, new = before[0], after[0]
    # Do not interpret a list or an unquoted multiword name as one object.
    continuation = re.compile(r'^\s*(?:[,/&+]|(?:и|или|and|or)\b|[A-ZА-ЯЁ])')
    if any(continuation.match(text[match.end():])
           for text, match in ((original, old), (current, new))):
        return action

    def context(text, match):
        return ' '.join((text[:match.start('name')] + '\0' +
                         text[match.end('name'):]).casefold().split())

    if context(original, old) != context(current, new):
        return action
    matches = list(service.finditer(action))
    if len(matches) != 1 or matches[0]['name'] != old['name']:
        return action
    target = matches[0]
    if continuation.match(action[target.end():]):
        return action
    return action[:target.start('name')] + new['name'] + action[target.end('name'):]


class Recall:
    def __init__(self, model):
        self.model = model
        self._state = dict(situations=[], decisions=[], outcomes=[], experiences=[])

    def snapshot(self):
        return deepcopy(self._state)

    def decide(self, text):
        if not isinstance(text, str) or not 1 <= len(text.strip()) <= 2000:
            raise ValueError('Введите ситуацию длиной от 1 до 2000 символов.')
        text = text.strip()
        features = features_for(text)
        if not features or len(features) > 200:
            raise ValueError('Ситуация должна содержать от 1 до 200 разных слов или чисел.')
        if len(self._state['decisions']) >= 100:
            raise ValueError('Достигнут предел 100 решений в этой сессии.')
        candidates = []
        for exp in self._state['experiences']:
            score = similarity(features, exp['situation_features'])
            if exp['success'] and exp['score'] >= SUCCESS_SCORE_THRESHOLD and score >= TRANSFER_THRESHOLD:
                candidates.append((score, exp))
        evidence = None
        if candidates:
            score, exp = min(candidates, key=lambda item: (
                -item[0], -item[1]['score'], normalize(item[1]['action']), item[1]['id']))
            original = next((s['text'] for s in self._state['situations']
                             if s['id'] == exp['situation_id']), '')
            action = adapt_transferred_action(exp['action'], original, text)
            reason = f"Использован подтверждённый опыт №{exp['id']}. Сходство: {score:.0%}."
            source = 'EXPERIENCE'
            evidence = dict(experience=deepcopy(exp), similarity=score,
                            score=exp['score'], action=action)
        else:
            # Validate even injected model adapters, before any state mutation.
            from model_client import validate_output
            answer = validate_output(self.model.decide(text, features), features)
            features, action, reason = answer['features'], answer['action'], answer['reason']
            source = 'MODEL'
        identifier = len(self._state['decisions']) + 1
        situation = dict(id=identifier, text=text, features=features)
        decision = dict(id=identifier, situation_id=identifier, action=action,
                        source=source, reason=reason, evidence=evidence)
        self._state['situations'].append(situation)
        self._state['decisions'].append(decision)
        return deepcopy(decision)

    def save_outcome(self, decision_id, success, score, feedback):
        if type(decision_id) is not int or not 1 <= decision_id <= len(self._state['decisions']):
            raise ValueError('Выберите существующее решение.')
        if type(success) is not bool:
            raise ValueError('Успех должен быть отмечен или снят.')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Оценка должна быть числом от 0 до 1.')
        if not isinstance(feedback, str) or len(feedback) > 1000:
            raise ValueError('Обратная связь: не более 1000 символов.')
        if any(o['decision_id'] == decision_id for o in self._state['outcomes']):
            raise ValueError('Исход этого решения уже сохранён.')
        decision = self._state['decisions'][decision_id - 1]
        situation = self._state['situations'][decision['situation_id'] - 1]
        outcome = dict(decision_id=decision_id, success=success, score=float(score), feedback=normalize(feedback))
        experience = dict(id=len(self._state['experiences']) + 1,
                          situation_id=situation['id'], decision_id=decision_id,
                          situation_features=deepcopy(situation['features']),
                          action=decision['action'], success=success, score=float(score))
        self._state['outcomes'].append(outcome)
        self._state['experiences'].append(experience)
        return deepcopy(experience)
