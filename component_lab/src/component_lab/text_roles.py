"""Conservative schematic OCR roles. Candidates are evidence, never OCR truth.

Original strings are kept verbatim. In particular O/0 and I/l/1 alternatives
are recorded, not silently substituted into reference or part-number fields.
"""
from __future__ import annotations

import re
from copy import deepcopy


PREFIX_TYPES = {
    "R": {"resistor", "potentiometer"}, "RV": {"potentiometer", "resistor"},
    "VR": {"potentiometer", "voltage_regulator"}, "C": {"capacitor"},
    "L": {"inductor"}, "FB": {"inductor", "ferrite_bead"},
    "D": {"diode", "led", "zener_diode", "esd_protection"}, "LED": {"led", "diode"},
    "Q": {"transistor", "bjt", "mosfet", "jfet"},
    "U": {"ic", "opamp", "optocoupler", "voltage_regulator", "logic_gate"},
    "IC": {"ic", "opamp", "optocoupler", "logic_gate"},
    "J": {"connector"}, "P": {"connector"}, "CN": {"connector"}, "CON": {"connector"},
    "JP": {"jumper", "connector"}, "SW": {"switch"}, "TP": {"test_point"},
    "F": {"fuse"}, "Y": {"crystal", "oscillator"}, "X": {"crystal", "connector"},
    "K": {"relay"}, "BT": {"battery"}, "BAT": {"battery"},
    "T": {"transformer", "transistor"}, "M": {"motor"}, "ANT": {"antenna"},
    "V": {"voltage_current_source"}, "I": {"voltage_current_source"},
}
REFERENCE = re.compile(r"^(" + "|".join(sorted(PREFIX_TYPES, key=len, reverse=True)) + r")(\d+)([A-Z]?)$", re.I)
NUMERIC_UNIT_REFERENCE = re.compile(r"^(U|IC)(\d+)\.(\d{1,2})$", re.I)
VALUE = re.compile(
    r"^(?:[+-]?\d+(?:\.\d+)?\s*(?:[pnumkKMGTµμ]?(?:F|H|Ω|ohms?|Hz|V|A|W)|[kKMGT])"
    r"|\d+[RrKkMm]\d*|\d+(?:\.\d+)?\s*%|\d+(?:\.\d+)?\s*V\s*/\s*\d+(?:\.\d+)?\s*A)$"
)
NUMERIC = re.compile(r"^\d+(?:\.\d+)?$")
NETWORK = re.compile(r"^(?:[+-]?(?:\d+(?:\.\d+)?V\d*|VCC\w*|VDD\w*|VSS\w*|VEE\w*)|[AD]?GND\w*|GNDA|GNDD|DGND|AGND|PE|EARTH|SDA|SCL|MOSI|MISO|SCK|SCLK|CLK|RESET|RST|NRST|TX\w*|RX\w*|GPIO\w*|USB[_+-]?[DP]\w*)$", re.I)
PIN_NAME = re.compile(r"^(?:IN[+-]?|OUT\w*|EN|ENABLE|NC|NO|COM|VIN|VOUT|GATE|SOURCE|DRAIN|BASE|EMITTER|COLLECTOR|CS|CE|OE|A\d+|D\d+)$", re.I)
PART = re.compile(r"^(?=.{4,48}$)(?=.*[A-Za-z])(?=.*\d)[A-Za-z][A-Za-z0-9_.+\-/]*$")
SIGNAL = re.compile(r"^(?:(?:BOOT|COMM|CLAMP|I[O0]|PI[O0]|P[IO]|GPI[O0]|J?AC|CS|RESET|SIO|FLASH|RTC|WP|SO|SI|MISO|MOSI|PA|PB|PC)\d*|[A-Za-z0-9]+_[A-Za-z0-9_]+|[A-Za-z0-9]+/(?:SIO|IO)\d+|VCC[-_]\w+)$", re.I)


def value_kind(value):
    token = re.sub(r"\s+", "", value)
    if token.endswith("%"):
        return "tolerance"
    if re.search(r"Hz$", token, re.I):
        return "frequency"
    if re.search(r"F$", token):
        return "capacitance"
    if re.search(r"H$", token):
        return "inductance"
    if re.search(r"(?:Ω|ohms?)$", token, re.I) or re.fullmatch(r"\d+(?:\.\d+)?[kKM]|\d+[RrKkMm]\d*", token):
        return "resistance"
    if re.search(r"V$", token):
        return "voltage"
    if re.search(r"A$", token):
        return "current"
    if re.search(r"W$", token):
        return "power"
    return "numeric" if NUMERIC.fullmatch(token) else "other"


def _candidate(role, value, score, reason, **extra):
    return {"role": role, "value": value, "heuristic_score": score,
            "reason": reason, "requires_review": False, **extra}


def parse_text_role(text, *, numeric_unit_refs=False):
    """Return a copy with ``role_candidates``; scores are uncalibrated rules.

    Pure numbers primarily remain pin candidates. A low-rank value candidate
    explicitly requires nearby reference + compatible component context.
    """
    row = deepcopy(text)
    raw = str(row.get("raw", row.get("text", "")))
    token = raw.strip()
    compact = re.sub(r"\s+", "", token)
    row.update(raw=raw, text=raw, normalized_whitespace=token)
    candidates = []
    if "\t" in raw:
        row.update(role_candidates=[_candidate("description", raw, .1,
                                               "control_characters_require_ocr_review", requires_review=True)],
                   primary_role="description", role_score_kind="uncalibrated_rule_evidence")
        return row
    match = REFERENCE.fullmatch(compact)
    numeric_unit = NUMERIC_UNIT_REFERENCE.fullmatch(compact) if numeric_unit_refs and not match else None
    match = match or numeric_unit
    if match:
        prefix, number, unit = match.groups()
        prefix, unit = prefix.upper(), unit.upper()
        reference = prefix + number + ('.' if numeric_unit else '') + unit
        physical = prefix + number if prefix in {"U", "IC", "Q"} else reference
        candidates.append(_candidate("reference", reference, .94, "numeric_unit_reference_syntax" if numeric_unit else "reference_syntax",
                                     prefix=prefix, physical_reference=physical,
                                     unit=unit or None,
                                     whitespace_compacted=compact != token))
    elif compact:
        # Only record a small, explicit candidate set. Never correct a model.
        guessed = compact.translate(str.maketrans({"l": "1", "I": "1", "O": "0"}))
        alternative = REFERENCE.fullmatch(guessed)
        if guessed != compact and alternative:
            candidates.append(_candidate("reference", guessed.upper(), .40,
                                         "confusable_character_alternative", requires_review=True,
                                         alternative_to=raw, prefix=alternative.group(1).upper()))
    if VALUE.fullmatch(token) or VALUE.fullmatch(compact):
        candidates.append(_candidate("value", token, .90, "explicit_electrical_unit_or_rkm_value",
                                     value_kind=value_kind(token)))
    if NUMERIC.fullmatch(token):
        candidates.append(_candidate("pin", token, .72, "pure_number_not_unconditional_value"))
        candidates.append(_candidate("value", token, .30, "numeric_value_requires_reference_context",
                                     requires_context=True, value_kind="numeric",
                                     requires_review=bool(re.fullmatch(r"[1-9]", token))))
    if NETWORK.fullmatch(compact):
        candidates.append(_candidate("net", token, .87, "network_or_power_label_syntax"))
    if PIN_NAME.fullmatch(compact):
        candidates.append(_candidate("pin", token, .68, "pin_name_syntax"))
        candidates.append(_candidate("net", token, .50, "pin_name_can_be_network_label"))
    signal = SIGNAL.fullmatch(compact)
    if signal:
        candidates.append(_candidate("net", token, .75, "signal_label_candidate"))
        candidates.append(_candidate("pin", token, .70, "signal_can_be_pin_name"))
    punctuated_ref = REFERENCE.fullmatch(compact.rstrip(".,:;"))
    metadata_or_partial = bool(re.fullmatch(r"V\d+\.\d+|[pnumkMµμ]?[FH]/\d+V|(?:SC|SOT|SOP|QFN|LQFP)\d+", compact, re.I))
    if PART.fullmatch(compact) and not match and not NETWORK.fullmatch(compact) and not signal and not punctuated_ref and not metadata_or_partial:
        composite = bool(re.search(r"\s", token) or re.match(r"^[RCLDUJ]\d{1,3}[A-Z]{2,}\d", compact))
        candidates.append(_candidate("part_number", token, .70, "merged_text_model_requires_review" if composite else "alphanumeric_model_candidate",
                                     requires_review=composite))
    if not candidates:
        candidates.append(_candidate("description", token, .35, "unresolved_free_text"))
    candidates.sort(key=lambda item: -item["heuristic_score"])
    row["role_candidates"] = candidates
    row["primary_role"] = candidates[0]["role"]
    row["role_score_kind"] = "uncalibrated_rule_evidence"
    return row


def parse_texts(texts, *, numeric_unit_refs=False):
    return [parse_text_role({**row, "id": row.get("id", f"text_{i:04d}")}, numeric_unit_refs=numeric_unit_refs)
            for i, row in enumerate(texts, 1)]
