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


def critical_mismatch(action, original, current):
    """Detect explicit action prohibitions/unavailability, not general semantics.

    Supported local clauses: 'откат [версии] недоступен',
    'недоступен откат [версии]', 'нельзя откатить ...'.
    The vocabulary describes constraints, never particular services or objects.
    """
    negative = r'(?:не\s*доступ(?:ен|на|но|ны)|не\s*возмож(?:ен|на|но|ны)|запрещ[её]н(?:а|о|ы)?)'

    def stem(word):
        return re.sub(r'(?:ить|ать|ять|ы|а|у|ом|е)$', '', word)

    operators = {'выполнить', 'сделать', 'провести', 'осуществить'}

    def english_terms(text):
        return re.sub(r'\broll\s+back\b', 'rollback', text.casefold())

    def restrictions(text):
        result = {}
        for clause in re.split(r'[.!?;\n,]+', normalize(text).casefold()):
            clause = clause.strip()
            if re.search(r'\bне\s+(?:запрещ[её]н\w*|невозмож\w*|недоступ\w*)\b', clause):
                continue
            # Anchoring avoids turning 'не запрещён' into a prohibition.
            match = re.fullmatch(r'(?:нельзя|невозможно|запрещено)\s+'
                                 r'(?:(?:выполнить|сделать|провести|осуществить)\s+)?'
                                 r'(\w+)(?:\s+.+)?', clause)
            if not match:
                match = re.fullmatch(rf'{negative}\s+(\w+)(?:\s+[^.!?;]+)?', clause)
            if not match:
                match = re.fullmatch(rf'(\w+)(?:\s+\w+){{0,2}}\s+{negative}', clause)
            if match and match[1] not in {'не', 'нет', 'если'}:
                result[stem(match[1])] = clause
            english = english_terms(clause)
            constraint = r'(?:unavailable|not available|impossible|not possible|prohibited|forbidden|disabled|not permitted|not allowed)'
            match = re.fullmatch(rf'(?:the\s+)?(\w+)(?:\s+(?:is|are))?\s+{constraint}', english)
            if not match:
                match = re.fullmatch(r'(?:do not|cannot|can not|must not)\s+'
                                     r'(?:(?:perform|execute)\s+)?(\w+)(?:\s+.+)?', english)
            if match:
                result[stem(match[1])] = clause
        return result

    previous = restrictions(original)
    action_words = {stem(word) for word in features_for(english_terms(action)) if word not in operators}
    for subject, clause in restrictions(current).items():
        if subject in action_words and subject not in previous:
            return f'New constraint on the action: “{clause}”.'
    return None


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
            raise ValueError('Enter a situation between 1 and 2,000 characters long.')
        text = text.strip()
        features = features_for(text)
        if not features or len(features) > 200:
            raise ValueError('The situation must contain between 1 and 200 distinct words or numbers.')
        if len(self._state['decisions']) >= 100:
            raise ValueError('This session has reached the limit of 100 decisions.')
        candidates = []
        for exp in self._state['experiences']:
            score = similarity(features, exp['situation_features'])
            if exp['success'] and exp['score'] >= SUCCESS_SCORE_THRESHOLD and score >= TRANSFER_THRESHOLD:
                candidates.append((score, exp))
        evidence = None
        transferred = False
        if candidates:
            score, exp = min(candidates, key=lambda item: (
                -item[0], -item[1]['score'], normalize(item[1]['action']), item[1]['id']))
            original = next((s['text'] for s in self._state['situations']
                             if s['id'] == exp['situation_id']), '')
            mismatch = critical_mismatch(exp['action'], original, text)
            if mismatch:
                evidence = dict(experience=deepcopy(exp), similarity=score,
                                score=exp['score'], status='TRANSFER BLOCKED: critical mismatch',
                                reason=mismatch)
            else:
                action = adapt_transferred_action(exp['action'], original, text)
                reason = f"Reused confirmed experience #{exp['id']}. Similarity: {score:.0%}."
                source = 'EXPERIENCE'
                evidence = dict(experience=deepcopy(exp), similarity=score,
                                score=exp['score'], action=action)
                transferred = True
        if not transferred:
            # Validate even injected model adapters, before any state mutation.
            from model_client import ModelError, validate_output
            try:
                answer = validate_output(self.model.decide(text, features), features)
            except ModelError as exc:
                if evidence and evidence.get('status'):
                    raise ModelError(f"{evidence['status']}. {evidence['reason']} {exc}") from exc
                raise
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
            raise ValueError('Select an existing decision.')
        if type(success) is not bool:
            raise ValueError('Success must be either checked or unchecked.')
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('Enter a numeric score between 0 and 1.')
        if not isinstance(feedback, str) or len(feedback) > 1000:
            raise ValueError('Feedback must be no longer than 1,000 characters.')
        if any(o['decision_id'] == decision_id for o in self._state['outcomes']):
            raise ValueError('An outcome has already been saved for this decision.')
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
