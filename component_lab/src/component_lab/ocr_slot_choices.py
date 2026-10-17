"""Strict OCR-ID choices. Adapted from reviewed DeepSeek code suggestions.

The suggestion incorrectly used integer IDs and did not parse its JSON string;
both contracts are corrected here. No generated code/tools execute at runtime.
"""
import json


def _unique_object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError('Duplicate JSON key')
        out[key] = value
    return out


def parse_id_choices(content, component_ids, tokens, allowed):
    """Return fullpage_identity-compatible items, preserving literal OCR strings."""
    data = json.loads(content, object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or set(data) != {'items'} or not isinstance(data['items'], list):
        raise ValueError('Expected items object')
    if not all(isinstance(c, str) and c for c in component_ids) or len(set(component_ids)) != len(component_ids):
        raise ValueError('Component IDs must be unique strings')
    expected, seen, result = set(component_ids), set(), []
    for item in data['items']:
        if not isinstance(item, list) or len(item) != 4:
            raise ValueError('Each item must have four entries')
        ident, *fields = item
        if not isinstance(ident, str) or ident not in expected or ident in seen:
            raise ValueError('Unknown/duplicate component ID')
        seen.add(ident)
        row = {'id': ident, 'ocr_ids': {}, 'uncertain_fields': []}
        for field, token_id in zip(('reference', 'Name', 'value'), fields):
            raw, oid = None, None
            if token_id == '?' and isinstance(token_id, str):
                row['uncertain_fields'].append(field)
            elif token_id is not None:
                if not isinstance(token_id, str) or token_id not in tokens or token_id not in allowed.get(ident, []):
                    raise ValueError('Unknown or out-of-scope OCR ID')
                raw, oid = tokens[token_id]['raw'], token_id
                if not isinstance(raw, str):
                    raise ValueError('OCR raw must be a string')
            row[field], row['ocr_ids'][field] = raw, oid
        result.append(row)
    if seen != expected:
        raise ValueError('Missing component IDs')
    return result
