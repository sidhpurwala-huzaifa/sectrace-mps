"""Dataset adapters. Provenance never becomes model input.

Raw patch associations are not precise vulnerability labels. Imports preserve
that distinction, and no importer labels changed lines as evidence by default.
"""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlparse
from .schema import validate_record
from .util import atomic_json, digest, json_records, sha256_file, write_jsonl


def repo_identity(value):
    value = str(value or "").strip().rstrip("/")
    if value.endswith(".git"):
        value = value[:-4]
    if value.startswith("git@github.com:"):
        value = "github.com/" + value.split(":", 1)[1]
    if "://" in value:
        parsed = urlparse(value)
        value = parsed.netloc + parsed.path
    # Public code sources here use GitHub owner/repo shorthand. Normalize it
    # against full GitHub URLs so repository holdouts cross dataset boundaries.
    if re.fullmatch(r"[^/:]+/[^/:]+", value):
        value = "github.com/" + value
    return value.lower()


def cwes(value):
    return sorted(set(re.findall(r"CWE-\d+", json.dumps(value))))


def binary(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if str(value).lower() in ("0", "false"):
        return 0
    if str(value).lower() in ("1", "true"):
        return 1
    raise ValueError(f"Expected a binary label, received {value!r}")


def base(code, source, repo, group, identifier=None, **kwargs):
    if not group:
        raise ValueError(f"{source}: cannot establish incident/commit/repository grouping")
    return {"id": str(identifier or f"{source}:{digest(code)}"), "code": code,
            "source": source, "repo": repo_identity(repo), "group_id": str(group),
            "label": None, "language": "c", "weight": 1.0, **kwargs}


def primevul(r):
    if "func" not in r or "target" not in r:
        raise ValueError("PrimeVul adapter expects `func` and `target`; use original JSONL releases")
    repo = r.get("project_url") or r.get("repo_url") or r.get("project") or r.get("repo")
    commit = r.get("commit_id") or r.get("commit_hash")
    incident = r.get("cve") or r.get("cve_id")
    if isinstance(incident, dict):
        incident = incident.get("id") or incident.get("cve_id")
    if incident and not re.fullmatch(r"CVE-\d{4}-\d+", str(incident)):
        incident = None
    group = incident or (f"{repo_identity(repo)}@{commit}" if commit else repo_identity(repo))
    out = base(r["func"], "PrimeVul", repo, group,
               identifier=f"PrimeVul:{r.get('idx', r.get('func_hash', digest(r['func'])))}",
               label=binary(r["target"]), cwes=cwes(r.get("cwe", r.get("cwe_ids"))) if binary(r["target"]) == 1 else [],
               confidence="source_dataset_label", incident=incident, commit=commit,
               source_split=r.get("split"), split=r.get("split"),
               label_scope="source dataset function-level judgment")
    return [out]


def megavul(r):
    if "is_vul" not in r or "func" not in r:
        raise ValueError("MegaVul adapter expects flattened megavul(_simple).json")
    repo, commit, incident = r.get("repo_name"), r.get("commit_hash"), r.get("cve_id")
    group = incident or f"{repo_identity(repo)}@{commit}"
    common = dict(language="c++" if str(r.get("file_path", "")).endswith((".cpp", ".cc", ".cxx")) else "c",
                  incident=incident, commit=commit, published_date=r.get("publish_date"),
                  source_path=r.get("file_path"), confidence="mined_patch_association", weight=0.4)
    if binary(r["is_vul"]):
        if not isinstance(r.get("func_before"), str):
            raise ValueError("MegaVul vulnerable row has no func_before; refusing to mislabel post-fix func")
        before = base(r["func_before"], "MegaVul", repo, group,
                      label=1, cwes=cwes(r.get("cwe_ids")), **common)
        after = base(r["func"], "MegaVul", repo, group,
                     label=None, fixed_for_incident=incident,
                     review_required="Post-fix does not prove absence of other flaws", **common)
        return [before, after]
    return [base(r["func"], "MegaVul", repo, group, label=0,
                 label_scope="source dataset negative, not independently verified", **common)]


def secvuleval(r):
    # Official SecVulEval schema: func_body, is_vulnerable, cve_list, cwe_list.
    # context is teacher-generated/privileged; changed_statements are not truth.
    code = r.get("func_body")
    if not isinstance(code, str) or "is_vulnerable" not in r:
        raise ValueError("SecVulEval expects func_body and is_vulnerable; inspect the release schema")
    repo = r.get("project_url") or r.get("project")
    commit = r.get("commit_id")
    incidents = sorted(set(re.findall(r"CVE-\d{4}-\d+", json.dumps(r.get("cve_list", [])))))
    group = f"{repo_identity(repo)}@{commit}" if repo and commit else ("+".join(incidents) or repo_identity(repo))
    label = binary(r["is_vulnerable"])
    out = base(code, "SecVulEval", repo, group,
               identifier=f"SecVulEval:{r.get('idx', digest(code))}",
               label=label, cwes=cwes(r.get("cwe_list")) if label == 1 else [],
               incidents=incidents, incident=incidents[0] if incidents else None,
               commit=commit, source_path=r.get("filepath"),
               confidence="source_dataset_label", weight=0.5,
               omitted_privileged_context=True, evidence_requires_review=True)
    return [out]


def generic_code(r, source="local-code"):
    code = r.get("code", r.get("content"))
    if not isinstance(code, str):
        if "blob_id" in r or "swhid" in r or "content_id" in r:
            raise ValueError("This release contains content identifiers, not code. Hydrate through authorized upstream access first.")
        raise ValueError("Code import requires code or content")
    repo = r.get("repo") or r.get("repository_name") or r.get("repo_name") or r.get("repo_url")
    if not repo:
        raise ValueError("Code import needs repository identity for holdout exclusion")
    return [base(code, source, repo, repo_identity(repo), language=r.get("lang", r.get("language", "c")),
                 license=r.get("licenses", r.get("license", r.get("detected_licenses"))), source_path=r.get("path"),
                 commit=r.get("revision_id", r.get("commit_id")), snapshot_id=r.get("snapshot_id"),
                 upstream_content_id=r.get("content_id", r.get("blob_id")))]


def convert_file(input_path, output_path, format, split=None):
    adapters = {"primevul": primevul, "megavul": megavul, "secvuleval": secvuleval,
                "code": generic_code, "canonical": lambda r: [validate_record(r)]}
    stats = Counter()
    def records():
        for raw in json_records(input_path):
            stats["source_rows"] += 1
            for r in adapters[format](raw):
                if split:
                    r["split"] = split
                stats["emitted_rows"] += 1
                stats["labeled_rows"] += int(r.get("label") is not None)
                yield validate_record(r)
    write_jsonl(output_path, records())
    atomic_json(str(output_path) + ".manifest.json", {"source_file_sha256": sha256_file(input_path),
                "format": format, "stats": dict(stats), "license_note": "Underlying source licenses require separate review."})
    print(json.dumps(dict(stats), indent=2))


def import_hf(dataset, output, revision=None, data_dir=None, split="train", limit=20000,
              format="code", license_allowlist=None, acknowledge_terms=False):
    if not acknowledge_terms:
        raise ValueError("Review upstream access/licensing/removal terms, then pass --acknowledge-terms")
    if limit <= 0:
        raise ValueError("Use an explicit positive record limit to avoid accidental bulk downloads")
    from datasets import load_dataset
    from huggingface_hub import HfApi
    # Resolve and record a concrete revision before fetching any examples.
    resolved = HfApi().dataset_info(dataset, revision=revision).sha
    kwargs = {"split": split, "streaming": True, "revision": resolved}
    if data_dir:
        kwargs["data_dir"] = data_dir
    stream = load_dataset(dataset, **kwargs)  # no trust_remote_code, no gate bypass
    allowed = {s.strip().lower() for s in (license_allowlist or [])}
    if format == "code" and not allowed:
        raise ValueError("Code-corpus imports require --licenses with your approved license identifiers")
    stats = Counter()
    adapters = {"code": lambda r: generic_code(r, dataset), "primevul": primevul, "secvuleval": secvuleval}
    def records():
        for raw in stream:
            stats["scanned"] += 1
            if format == "code":
                values = raw.get("licenses", raw.get("license", raw.get("detected_licenses", [])))
                values = [values] if isinstance(values, str) else values
                if not isinstance(values, list):
                    stats["license_filtered"] += 1
                    continue
                values = {v.lower() for v in values if isinstance(v, str)}
                if not values or not values.issubset(allowed):
                    stats["license_filtered"] += 1
                    continue
                language = str(raw.get("lang", raw.get("language", ""))).lower()
                if language not in ("c", "c++", "cpp"):
                    stats["language_filtered"] += 1
                    continue
            for r in adapters[format](raw):
                r["source_revision"] = resolved
                r["upstream_split"] = split
                stats["emitted"] += 1
                yield validate_record(r)
                if stats["emitted"] >= limit:
                    return
    count = write_jsonl(output, records())
    if count == 0:
        raise ValueError("No rows passed schema/language/license filters; inspect the upstream release")
    atomic_json(str(output) + ".manifest.json", {"dataset": dataset, "revision": resolved,
                "data_dir": data_dir, "split": split, "stats": dict(stats), "approved_license_filter": sorted(allowed)})
    print(json.dumps(dict(stats), indent=2))


CVEFIXES_SQL = """
SELECT m.method_change_id AS id, m.code AS code, m.before_change AS before_change,
       f.hash AS commit_hash, f.programming_language AS language,
       fx.repo_url AS repo, fx.cve_id AS incident
FROM method_change m
JOIN file_change f ON f.file_change_id = m.file_change_id
JOIN fixes fx ON fx.hash = f.hash
WHERE lower(f.programming_language) IN ('c', 'c++', 'cpp')
"""


def import_sqlite(database, output, query_file=None):
    # SELECT from a read-only database; never execute dump scripts or generated SQL.
    query = Path(query_file).read_text() if query_file else CVEFIXES_SQL
    if not query.lstrip().lower().startswith("select"):
        raise ValueError("The adapter accepts SELECT queries only")
    connection = sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    connection.row_factory = sqlite3.Row
    def rows():
        try:
            for raw in connection.execute(query):
                r = dict(raw)
                for key in ("code", "repo", "incident"):
                    if key not in r:
                        raise ValueError(f"SQL must expose {key!r} as a column")
                # A method touched by a security fix is only a candidate.
                yield validate_record(base(r["code"], "CVEfixes-SQL", r["repo"], r["incident"],
                      identifier=f"CVEfixes:{r.get('id', digest(r['code']))}:{r['incident']}",
                      language=r.get("language", "c"), incident=r["incident"],
                      commit=r.get("commit_hash"), candidate_before_change=r.get("before_change"),
                      confidence="unreviewed_patch_candidate", weight=0.3,
                      review_required="Confirm affected method and label scope before SFT"))
        finally:
            connection.close()
    count = write_jsonl(output, rows())
    atomic_json(str(output) + ".manifest.json", {"source": "read-only SQLite export",
                "database_sha256": sha256_file(database), "rows": count,
                "query_sha256": digest(query), "default_labels": "unknown pending review"})
    print(f"Imported {count} unreviewed candidates; usable for pre-training, not yet binary SFT.")
