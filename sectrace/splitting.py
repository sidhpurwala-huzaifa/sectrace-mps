"""Global group splitting and explicit leakage audit; not a clone-proof guarantee."""
from __future__ import annotations
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from .importers import repo_identity
from .schema import validate_record
from .tokenization import lexical_fingerprint
from .util import atomic_json, digest, json_records


class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, key):
        self.parent.setdefault(key, key)
        root = key
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[key] != key:
            key, self.parent[key] = self.parent[key], root
        return root
    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        # Lexical root makes membership deterministic independent of input order.
        self.parent[max(a, b)] = min(a, b)


def identities(r, group_key="repo", aliases=None):
    aliases = aliases or {}
    group = str(r["group_id"])
    result = ["group:" + group]
    if r.get("incident"):
        result.append("incident:" + str(r["incident"]))
    for incident in r.get("incidents", []):
        result.append("incident:" + str(incident))
    if group_key == "repo":
        repo = repo_identity(r.get("repo"))
        if not repo:
            raise ValueError(f"{r['id']}: repository holdout requested but repository is unknown")
        result.append("repo:" + aliases.get(repo, repo))
    result.append("lex:" + lexical_fingerprint(r["code"]))
    if "mate" in r:
        result.extend(identities(r["mate"], group_key, aliases))
    return result


def split_corpus(inputs, output_dir, group_key="repo", seed=17, respect_splits=False, alias_file=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dbfile = out / "split_work.sqlite"
    if dbfile.exists() or (out / "split_manifest.json").exists():
        raise FileExistsError("Split directory already used; choose a new directory")
    aliases = json.loads(Path(alias_file).read_text()) if alias_file else {}
    uf, fingerprint_labels, explicit = UnionFind(), defaultdict(set), defaultdict(set)
    connection = sqlite3.connect(dbfile)
    connection.execute("CREATE TABLE records (rowid INTEGER PRIMARY KEY, anchor TEXT, fp TEXT, score INTEGER, content TEXT)")
    n = 0
    for path in inputs:
        for raw in json_records(path):
            r = validate_record(raw)
            keys = identities(r, group_key, aliases)
            for key in keys[1:]:
                uf.union(keys[0], key)
            fp = lexical_fingerprint(r["code"])
            if r.get("label") is not None:
                fingerprint_labels[fp].add(r["label"])
            if "mate" in r and r["mate"].get("label") is not None:
                fingerprint_labels[lexical_fingerprint(r["mate"]["code"])].add(r["mate"]["label"])
            if respect_splits and r.get("split"):
                explicit[keys[0]].add(r["split"])
            score = 100 * int("mate" in r) + 10 * int(r.get("label") is not None) + len(r.get("evidence", []))
            connection.execute("INSERT INTO records(anchor,fp,score,content) VALUES(?,?,?,?)",
                               (keys[0], fp, score, json.dumps(r, ensure_ascii=False)))
            n += 1
            if n % 2000 == 0:
                connection.commit()
    connection.commit()
    root_splits = defaultdict(set)
    for key, values in explicit.items():
        root_splits[uf.find(key)].update(values)
    for root, values in root_splits.items():
        if len(values) > 1:
            raise ValueError(f"Source partitions overlap under requested grouping: {values}. Do not silently reassign a benchmark test set.")
    conflicts = {fp for fp, values in fingerprint_labels.items() if len(values) > 1}
    handles = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8")
               for name in ("train", "validation", "calibration", "test", "quarantine")}
    seen, counts, components = set(), Counter(), Counter()
    try:
        for anchor, fp, _, content in connection.execute("SELECT anchor,fp,score,content FROM records ORDER BY score DESC,rowid"):
            r = json.loads(content)
            mate_fp = lexical_fingerprint(r["mate"]["code"]) if "mate" in r else None
            if fp in conflicts or mate_fp in conflicts:
                handles["quarantine"].write(content + "\n")
                counts["conflicting_labels_quarantined"] += 1
                continue
            if fp in seen:
                counts["lexical_duplicates_removed"] += 1
                continue
            seen.add(fp)
            root = uf.find(anchor)
            if root_splits.get(root):
                split = next(iter(root_splits[root]))
            else:
                p = int(digest(f"{seed}:{root}")[:16], 16) / 2**64
                split = "train" if p < .8 else "validation" if p < .9 else "calibration" if p < .95 else "test"
            if split not in handles or split == "quarantine":
                raise ValueError(f"Unrecognized partition {split}")
            r["split"] = split
            r["split_component"] = digest(root)
            if "mate" in r:
                r["mate"]["split"] = split
            handles[split].write(json.dumps(r, ensure_ascii=False) + "\n")
            counts[split] += 1
            components[root] += 1
    finally:
        for handle in handles.values():
            handle.close()
        connection.close()
    manifest = {"seed": seed, "group_key": group_key, "source_rows": n, "counts": dict(counts),
                "components": len(components), "largest_component_records": max(components.values(), default=0),
                "respect_source_splits": respect_splits,
                "limitations": ["Not a near-clone detector", "Repository alias map is user-supplied",
                                "Whole-record hashes do not detect a held-out function embedded in a larger file",
                                "Hash group split is not a chronological split"]}
    atomic_json(out / "split_manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


def audit_files(inputs, output=None, group_key="repo", alias_file=None):
    seen, issues = {}, []
    aliases = json.loads(Path(alias_file).read_text()) if alias_file else {}
    for path in inputs:
        for r in json_records(path):
            r = validate_record(r)
            split = r.get("split")
            if split not in ("train", "validation", "calibration", "test"):
                raise ValueError("Every audited record needs an explicit partition")
            for key in identities(r, group_key, aliases):
                old = seen.get(key)
                if old is not None and old[0] != split:
                    if len(issues) < 100:
                        issues.append({"kind": key.split(":", 1)[0], "partitions": [old[0], split],
                                       "example_ids": [old[1], r["id"]]})
                else:
                    seen[key] = (split, r["id"])
    report = {"passed": not issues, "examples_of_overlap": issues,
              "scope": "exact lexical fingerprint, declared case/group, and optional normalized repository identity; no semantic clone guarantee"}
    if output:
        atomic_json(output, report)
    print(json.dumps(report, indent=2))
    if issues:
        raise ValueError("Data leakage audit failed; resolve overlaps before training")
    return report
