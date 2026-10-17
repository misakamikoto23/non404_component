"""Conservative geometry baseline. Keep ambiguous text for human review."""
import re

REFERENCE = re.compile(r"^(?:R|C|L|D|Q|U|J|P|SW|TP|F|Y|X|LED|RV|VR|K|BT)\d+[A-Z]?$", re.I)
VALUE = re.compile(r"^(?:\d+(?:\.\d+)?\s*(?:[pnumkMµμ]?F|[pnumkMµμ]?H|[kM]?Ω|ohm|V|A|Hz)|\d+[RrKkMm]\d*|\d+(?:\.\d+)?)$", re.I)


def associate(components, texts, margin=60, ambiguity_gap=0.15):
    for component in components:
        component.update(reference=None, Name=None, value=None, texts=[], review=[])
    unassigned = []
    for text in texts:
        a, b, c, d = text["bbox"]
        tx, ty = (a+c)/2, (b+d)/2
        candidates = []
        for i, component in enumerate(components):
            x1, y1, x2, y2 = component["bbox"]
            dx, dy = max(x1-tx, 0, tx-x2), max(y1-ty, 0, ty-y2)
            distance = (dx*dx+dy*dy)**0.5
            if distance <= margin:
                candidates.append((distance/max(1, margin), i))
        candidates.sort()
        if not candidates or (len(candidates)>1 and candidates[1][0]-candidates[0][0] < ambiguity_gap):
            unassigned.append({**text, "reason": "no_nearby_component" if not candidates else "ambiguous_owner"})
            continue
        owner = components[candidates[0][1]]
        owner["texts"].append(text)
    for component in components:
        refs = [t["text"].strip() for t in component["texts"] if REFERENCE.fullmatch(t["text"].strip())]
        values = [t["text"].strip() for t in component["texts"] if VALUE.fullmatch(t["text"].strip())]
        if len(refs) == 1:
            component["reference"] = refs[0]
        elif refs:
            component["review"].append("multiple_reference_candidates")
        if len(values) == 1:
            component["value"] = values[0]
        elif values:
            component["review"].append("multiple_value_candidates")
        # Name can be an IC model in the official data. Do not equate it with reference.
        if component["reference"] is None:
            component["review"].append("reference_unresolved")
    refs = {}
    for component in components:
        if component["reference"]:
            refs.setdefault(component["reference"].upper(), []).append(component)
    for group in refs.values():
        if len(group) > 1:
            for component in group:
                component["reference"] = None
                component["review"].append("duplicate_reference")
    return unassigned
