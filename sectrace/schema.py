from __future__ import annotations
import math
from .util import digest

ROLES = ["origin", "transformation", "risky_operation", "guard", "allocation", "release", "use", "size_bounds"]
RELATIONS = ["ast_child", "ast_parent", "cfg_next", "cfg_previous", "definition_use", "use_definition",
             "control_true", "control_false", "argument_parameter", "parameter_argument",
             "return_use", "use_return", "memory_dependence", "memory_dependence_reverse", "self", "other"]
# Output slots, not a claim of coverage, taxonomy completeness, or official CWE grouping.
CWE_IDS = [20, 22, 23, 36, 59, 77, 78, 79, 89, 94, 119, 120, 121, 122, 125, 126,
           129, 131, 134, 170, 190, 191, 194, 195, 196, 197, 200, 209, 252, 253, 269, 276,
           285, 287, 295, 310, 319, 327, 362, 367, 369, 400, 401, 404, 415, 416, 426, 427,
           476, 502, 590, 601, 611, 617, 665, 681, 682, 704, 754, 755, 772, 787, 788, 908]
CWE = [f"CWE-{i}" for i in CWE_IDS]


def validate_record(record: dict) -> dict:
    r = dict(record)
    if not isinstance(r.get("code"), str) or not r["code"].strip():
        raise ValueError("A record needs nonempty source code in `code`")
    r.setdefault("id", digest(r["code"]))
    if not r.get("group_id"):
        raise ValueError(f"{r['id']}: group_id is mandatory; use incident, commit, or repository identity")
    r.setdefault("label", None)
    if r["label"] not in (None, 0, 1):
        raise ValueError("label must be 0, 1, or null")
    if r.get("adequate") not in (None, 0, 1):
        raise ValueError("adequate must be 0, 1, or null")
    if r.get("teacher_probability") is not None:
        p = float(r["teacher_probability"])
        if not math.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("teacher_probability must be finite and between 0 and 1")
    if not isinstance(r.get("incidents", []), list):
        raise ValueError("incidents must be a list")
    w = float(r.get("weight", 1.0))
    if not math.isfinite(w) or w <= 0:
        raise ValueError("weight must be finite and positive")
    r["weight"] = w
    a, b = r.get("target_span", [0, len(r["code"])])
    if not (0 <= a < b <= len(r["code"])):
        raise ValueError("target_span uses valid Unicode character offsets, end exclusive")
    r["target_span"] = [a, b]
    for e in r.get("evidence", []):
        if not 0 <= e["start"] < e["end"] <= len(r["code"]):
            raise ValueError("Evidence span out of bounds")
        for role, label in e.get("roles", {}).items():
            if role not in ROLES or label not in (0, 1):
                raise ValueError("Unknown evidence role or non-binary evidence label")
    if "mate" in r:
        if r.get("pair_relation") not in ("higher", "invariant") or not r.get("pair_verified", False):
            raise ValueError("Paired objectives require an explicitly verified relation")
        r["mate"] = validate_record(r["mate"])
        if "mate" in r["mate"]:
            raise ValueError("Nested mate records are not supported")
    return r
