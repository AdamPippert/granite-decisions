"""Strict JSON input and finite categorical outputs, independent of model text."""

import hashlib
import json
import math

MAX_BYTES = 262_144


class ContractError(ValueError):
    pass


def validate_json(value, depth=0):
    if depth > 32:
        raise ContractError("json_too_deep")
    if value is None or type(value) is bool:
        return
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeError:
            raise ContractError("invalid_unicode") from None
    elif type(value) in (int, float):
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            raise ContractError("nonfinite_number")
    elif type(value) is list:
        for item in value:
            validate_json(item, depth + 1)
    elif type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise ContractError("json_key_must_be_string")
            validate_json(key, depth + 1)
            validate_json(item, depth + 1)
    else:
        raise ContractError("not_json_data")


def dumps(value):
    validate_json(value)
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def loads(raw, limit=MAX_BYTES):
    if len(raw.encode() if isinstance(raw, str) else raw) > limit:
        raise ContractError("input_too_large")

    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ContractError("duplicate_json_key")
            result[key] = value
        return result

    try:
        value = json.loads(raw, object_pairs_hook=unique)
        validate_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        if isinstance(exc, ContractError):
            raise
        raise ContractError("invalid_json") from None


def digest(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def identifier(value):
    if type(value) is not str or not value.strip() or len(value) > 128:
        raise ContractError("invalid_identifier")
    return value


def is_description(value):
    """Descriptions may contain structured JSON; root scalars are not prompts."""
    return (type(value) is str and bool(value.strip()) or
            type(value) in (dict, list) and bool(value))


def questions(value):
    if type(value) is not dict or not 1 <= len(value) <= 64:
        raise ContractError("expected_1_to_64_questions")
    if len(dumps(value).encode()) > MAX_BYTES:
        raise ContractError("questions_too_large")
    clean = loads(dumps(value))
    for key, q in clean.items():
        identifier(key)
        if type(q) is not dict or set(q) - {"type", "instructions", "criteria"}:
            raise ContractError("invalid_question_fields")
        if not is_description(q.get("instructions")):
            raise ContractError("instructions_required")
        kind, criteria = q.get("type"), q.get("criteria")
        if kind == "choice":
            if type(criteria) is not dict or not 2 <= len(criteria) <= 255:
                raise ContractError("choice_requires_2_to_255_options")
            for label, description in criteria.items():
                identifier(label)
                if description is not None and not is_description(description):
                    raise ContractError("option_description_required")
        elif kind == "score":
            if type(criteria) is not list or not 2 <= len(criteria) <= 10 or not all(is_description(x) for x in criteria):
                raise ContractError("score_requires_2_to_10_described_levels")
        elif kind == "noul":
            if criteria is not None and (type(criteria) is not dict or set(criteria) != {"false", "true"} or not all(type(x) is str and x.strip() for x in criteria.values())):
                raise ContractError("noul_criteria_requires_false_and_true")
        else:
            raise ContractError("unknown_question_type")
    return clean


def state(value):
    if type(value) not in (dict, list, str):
        raise ContractError("state_must_be_object_array_or_string")
    if len(dumps(value).encode()) > MAX_BYTES:
        raise ContractError("state_too_large")
    return value


def decision_request(value, model_id):
    if (type(value) is not dict or not {"state", "questions"} <= set(value) or
            set(value) - {"state", "questions", "model"}):
        raise ContractError("request_requires_state_questions_and_optional_model")
    if "model" in value and (type(value["model"]) is not str or value["model"] != model_id):
        raise ContractError("requested_model_not_loaded")
    return value["state"], value["questions"]


def question_signature(q):
    """Match question content, without involving its caller-selected map key."""
    value = dict(q)
    if value["type"] == "noul" and value.get("criteria") is None:
        value.pop("criteria", None)
    return digest(value)


def labels(q):
    if q["type"] == "choice":
        return sorted(q["criteria"])
    if q["type"] == "score":
        return [str(i) for i in range(len(q["criteria"]))]
    return ["false", "true"]


def label_index(q, value):
    kind = q["type"]
    if kind == "noul":
        if type(value) is not bool:
            raise ContractError("noul_training_label_must_be_boolean")
        return int(value)
    if kind == "score":
        if type(value) is not int or not 0 <= value < len(q["criteria"]):
            raise ContractError("score_training_label_must_be_level_index")
        return value
    if type(value) is not str or value not in q["criteria"]:
        raise ContractError("unknown_training_label")
    return labels(q).index(value)


def typed_answer(q, probabilities):
    names = labels(q)
    p = [float(v) for v in probabilities]
    if len(p) != len(names) or not all(math.isfinite(v) and 0 <= v <= 1 for v in p) or not math.isclose(sum(p), 1.0, abs_tol=1e-6):
        raise ContractError("invalid_probability_distribution")
    if q["type"] == "noul":
        return {"type": "noul", "noul": p[1]}
    # Our documented definition, not a claim about Jev's confidence formula.
    entropy = -sum(v * math.log(v) for v in p if v > 0)
    answer = {"type": q["type"], "probabilities": dict(zip(names, p)),
              "confidence": min(1.0, max(0.0, 1 - entropy / math.log(len(p))))}
    if q["type"] == "choice":
        answer["choice"] = names[max(range(len(p)), key=p.__getitem__)]
    else:
        answer["score"] = sum(i * v for i, v in enumerate(p))
        answer["legend"] = dict(zip(names, q["criteria"]))
    return answer
