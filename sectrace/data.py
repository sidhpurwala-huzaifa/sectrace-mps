from __future__ import annotations
import bisect
from collections import Counter
import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from .schema import CWE, RELATIONS, ROLES, validate_record
from .tokenization import CLS, MASK, PAD, SEP, CodeTokenizer, token_kinds
from .util import atomic_json, digest, json_records, sha256_file


def map_graph(record, offsets, max_nodes=256, max_neighbors=16, token_fanout=4):
    raw = record.get("graph")
    if not raw or not raw.get("nodes"):
        return None, {}
    candidates = []
    for i, n in enumerate(raw["nodes"]):
        a, b = n["start"], n["end"]
        if not 0 <= a < b <= len(record["code"]):
            raise ValueError("Graph node source span is invalid")
        covered = [j for j, (x, y) in enumerate(offsets) if y > a and x < b and y > x]
        if covered and offsets[covered[0]][0] <= a and offsets[covered[-1]][1] >= b:
            candidates.append((i, covered[0], covered[-1] + 1))
    # Prefer the smallest source-anchored expressions; caps are explicit metadata.
    candidates.sort(key=lambda x: (x[2] - x[1], x[1], x[0]))
    chosen = candidates[:max_nodes]
    remap = {old: i for i, (old, _, _) in enumerate(chosen)}
    if not chosen:
        return None, {"graph_nodes_dropped": len(raw["nodes"])}
    spans = [[a, b] for _, a, b in chosen]
    neighbors = [[] for _ in chosen]
    edges = []
    dropped = 0
    seen = set()
    for edge in raw.get("edges", []):
        src, dst = int(edge["src"]), int(edge["dst"])
        rel = edge["type"]
        rel = RELATIONS.index(rel) if isinstance(rel, str) else int(rel)
        if not 0 <= rel < len(RELATIONS):
            raise ValueError("Unknown graph relation")
        if src not in remap or dst not in remap:
            dropped += 1
            continue
        s, d = remap[src], remap[dst]
        if (s, d, rel) in seen:
            continue
        seen.add((s, d, rel))
        # Message follows src -> dst: dst attends to src.
        if len(neighbors[d]) >= max_neighbors - 1:
            dropped += 1
            continue
        neighbors[d].append([s, rel])
        label = edge.get("evidence")
        if label not in (None, 0, 1):
            raise ValueError("Edge evidence must be 0, 1, or null")
        edges.append([s, d, rel, -1 if label is None else label])
    for i, nbr in enumerate(neighbors):
        nbr.insert(0, [i, 14])
    token_nodes = [[] for _ in offsets]
    overlaps_dropped = 0
    for i, (a, b) in enumerate(spans):
        for t in range(a, b):
            if len(token_nodes[t]) < token_fanout:
                token_nodes[t].append(i)
            else:
                overlaps_dropped += 1
    return {"spans": spans, "neighbors": neighbors, "token_nodes": token_nodes,
            "edges": edges}, {"graph_nodes_dropped": len(raw["nodes"]) - len(chosen),
                              "graph_edges_dropped": dropped, "graph_overlaps_dropped": overlaps_dropped}


def encode_record(r, tokenizer, max_length, pretrain=False, max_nodes=256, max_neighbors=16):
    r = validate_record(r)
    ids, offsets = tokenizer.encode(r["code"])
    if not ids:
        return []
    if not pretrain and len(ids) + 2 > max_length:
        return []   # NO inherited function labels on an arbitrary truncated prefix.
    windows = range(0, len(ids), max_length - 2) if pretrain else [0]
    kinds_all = token_kinds(r["code"], offsets)
    records = []
    for start in windows:
        end = min(len(ids), start + max_length - 2)
        sub = offsets[start:end]
        off = [(0, 0)] + list(sub) + [(0, 0)]
        a, b = r["target_span"]
        target = [False] + [y > a and x < b for x, y in sub] + [False]
        if not any(target) and not pretrain:
            continue
        roles = [0] + [1 if v else 2 for v in target[1:-1]] + [0]
        cwe = [-1.0] * len(CWE)
        if not pretrain:
            if r.get("cwe_complete", False):
                cwe = [0.0] * len(CWE)
            for known in r.get("known_cwes", []):
                if known in CWE:
                    cwe[CWE.index(known)] = 0.0
            for positive in r.get("cwes", []):
                if positive in CWE:
                    cwe[CWE.index(positive)] = 1.0
        evidence = [[-1.0] * len(ROLES) for _ in off]
        if not pretrain:
            for ann in r.get("evidence", []):
                for t, (x, y) in enumerate(off):
                    if y > ann["start"] and x < ann["end"] and y > x:
                        for role, value in ann["roles"].items():
                            evidence[t][ROLES.index(role)] = float(value)
        graph, diagnostics = map_graph(r, off, max_nodes, max_neighbors)
        family = [-1.0] * 12
        if not pretrain:
            for k, v in r.get("family_labels", {}).items():
                if not 0 <= int(k) < 12 or v not in (0, 1):
                    raise ValueError("Family labels use integer slots 0..11 and binary values")
                family[int(k)] = float(v)
        obj = {"id": r["id"] + (f":window{start}" if pretrain else ""),
               "group_id": r["group_id"], "repo": r.get("repo"), "split": r.get("split"),
               "code_hash": digest(r["code"]), "input_ids": [CLS] + ids[start:end] + [SEP],
               "offsets": off, "kind_ids": [0] + kinds_all[start:end] + [0],
               "role_ids": roles, "language": 1 if str(r.get("language", "c")).lower() in ("c++", "cpp") else 0,
               "target_mask": target, "label": -1 if pretrain or r["label"] is None else r["label"],
               "adequate": -1 if pretrain or r.get("adequate") is None else r["adequate"],
               "weight": r["weight"], "cwe": cwe, "family": family, "roles": evidence,
               "teacher_probability": -1 if pretrain or r.get("teacher_probability") is None else r["teacher_probability"],
               "graph": graph, "diagnostics": diagnostics,
               "input_was_windowed": pretrain and len(ids) + 2 > max_length}
        if not pretrain and "mate" in r:
            mates = encode_record(r["mate"], tokenizer, max_length, False, max_nodes, max_neighbors)
            if not mates:
                continue
            obj["mate"] = mates[0]
            obj["pair_relation"] = r["pair_relation"]
        records.append(obj)
    return records


def prepare(input_path, tokenizer_path, output_dir, stage, max_length=512, max_nodes=256, max_neighbors=16, purpose=None):
    if max_length < 8 or max_length % 8:
        raise ValueError("max_length must be at least 8 and divisible by 8")
    out = Path(output_dir)
    if (out / "manifest.json").exists() or (out / "tokens.jsonl").exists():
        raise FileExistsError(f"Refusing to overwrite cache {out}; use a new directory")
    out.mkdir(parents=True, exist_ok=True)
    tokenizer = CodeTokenizer(tokenizer_path)
    offsets, lengths = [], []
    stats = Counter()
    splits = set()
    with (out / "tokens.jsonl").open("wb") as f:
        for r in json_records(input_path):
            stats["input_records"] += 1
            if purpose and r.get("split") and purpose != r["split"]:
                raise ValueError("Cannot relabel a source partition through --purpose")
            if stage != "pretrain" and not any([r.get("label") is not None, r.get("teacher_probability") is not None, r.get("adequate") is not None, r.get("evidence"), r.get("cwes"), r.get("known_cwes"), r.get("family_labels"), r.get("mate"), any(e.get("evidence") is not None for e in (r.get("graph") or {}).get("edges", []))]):
                stats["rejected_no_supervision"] += 1
                continue
            if r.get("split"):
                splits.add(r["split"])
            encoded = encode_record(r, tokenizer, max_length, stage == "pretrain", max_nodes, max_neighbors)
            if not encoded:
                stats["rejected_overlength_or_empty"] += 1
            for item in encoded:
                offsets.append(f.tell())
                row = json.dumps(item, ensure_ascii=False, separators=(",", ":")).encode("utf-8") + b"\n"
                f.write(row)
                lengths.append(max(len(item["input_ids"]), len(item.get("mate", {}).get("input_ids", []))))
                stats["output_records"] += 1
                stats["tokens"] += len(item["input_ids"])
                stats["labeled_records"] += int(item["label"] >= 0)
                stats["graph_records"] += int(item["graph"] is not None)
                stats["paired_records"] += int("mate" in item)
        offsets.append(f.tell())
    if not lengths:
        raise ValueError("No records fit. Inspect input and increase max_length; no labels were silently truncated.")
    np.save(out / "offsets.npy", np.asarray(offsets, dtype=np.uint64))
    np.save(out / "lengths.npy", np.asarray(lengths, dtype=np.int32))
    if purpose is None:
        purpose = next(iter(splits)) if len(splits) == 1 else "unspecified"
    manifest = {"format": 1, "stage": stage, "purpose": purpose, "source_splits": sorted(splits),
                "max_length": max_length, "tokenizer_sha256": sha256_file(tokenizer_path),
                "input_sha256": sha256_file(input_path), "data_sha256": sha256_file(out / "tokens.jsonl"),
                "vocab_size": tokenizer.vocab_size, "random_vocab_size": tokenizer.random_vocab_size,
                "diagnostic_tokenizer": tokenizer.byte, "stats": dict(stats)}
    atomic_json(out / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2))
    return manifest


class CachedDataset:
    def __init__(self, directory):
        self.path = Path(directory)
        self.manifest = json.loads((self.path / "manifest.json").read_text())
        if sha256_file(self.path / "tokens.jsonl") != self.manifest["data_sha256"]:
            raise ValueError("Cache bytes no longer match the recorded fingerprint; rebuild the cache")
        self.offsets = np.load(self.path / "offsets.npy", mmap_mode="r")
        self.lengths = np.load(self.path / "lengths.npy", mmap_mode="r")
        self.file = (self.path / "tokens.jsonl").open("rb")

    def __len__(self):
        return len(self.lengths)

    def __getitem__(self, i):
        self.file.seek(int(self.offsets[i]))
        return json.loads(self.file.read(int(self.offsets[i + 1] - self.offsets[i])))

    def close(self):
        self.file.close()


class BatchStream:
    """Length-bucketed epochs, exact cursor resume, no MPS-unsafe worker forking."""
    def __init__(self, dataset, batch_size, seed, state=None):
        self.dataset, self.batch_size, self.seed = dataset, batch_size, seed
        self.epoch, self.cursor = (state["epoch"], state["cursor"]) if state else (0, 0)
        self._build()

    def _build(self):
        rng = random.Random(self.seed + self.epoch)
        order = list(range(len(self.dataset)))
        rng.shuffle(order)
        buckets = []
        width = max(self.batch_size, self.batch_size * 100)
        for start in range(0, len(order), width):
            block = sorted(order[start:start + width], key=lambda i: self.dataset.lengths[i])
            buckets.extend(block[i:i + self.batch_size] for i in range(0, len(block), self.batch_size))
        rng.shuffle(buckets)
        self.batches = buckets

    def next(self):
        if self.cursor == len(self.batches):
            self.epoch += 1
            self.cursor = 0
            self._build()
        indices = self.batches[self.cursor]
        self.cursor += 1
        return [self.dataset[i] for i in indices]

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}


def collate_graph(rows, length):
    graphs = [r.get("graph") for r in rows]
    if not any(g is not None for g in graphs):
        return None
    b = len(rows)
    n = max(1, max(len(g["spans"]) if g else 0 for g in graphs))
    k = max(1, max((len(v) for g in graphs if g for v in g["neighbors"]), default=0))
    fanout = max(1, max((len(v) for g in graphs if g for v in g["token_nodes"]), default=0))
    e = max(1, max(len(g["edges"]) if g else 0 for g in graphs))
    result = {"spans": torch.zeros(b, n, 2, dtype=torch.long),
              "node_mask": torch.zeros(b, n, dtype=torch.bool),
              "neighbors": torch.zeros(b, n, k, dtype=torch.long),
              "relations": torch.zeros(b, n, k, dtype=torch.long),
              "neighbor_mask": torch.zeros(b, n, k, dtype=torch.bool),
              "token_nodes": torch.zeros(b, length, fanout, dtype=torch.long),
              "token_node_mask": torch.zeros(b, length, fanout, dtype=torch.bool),
              "edge_index": torch.zeros(b, e, 2, dtype=torch.long),
              "edge_relations": torch.zeros(b, e, dtype=torch.long),
              "edge_targets": torch.full((b, e), -1.0)}
    for i, g in enumerate(graphs):
        if g is None:
            continue
        count = len(g["spans"])
        result["spans"][i, :count] = torch.tensor(g["spans"])
        result["node_mask"][i, :count] = True
        for j, nbr in enumerate(g["neighbors"]):
            if nbr:
                z = torch.tensor(nbr)
                result["neighbors"][i, j, :len(nbr)] = z[:, 0]
                result["relations"][i, j, :len(nbr)] = z[:, 1]
                result["neighbor_mask"][i, j, :len(nbr)] = True
        for j, nodes in enumerate(g["token_nodes"]):
            if nodes:
                result["token_nodes"][i, j, :len(nodes)] = torch.tensor(nodes)
                result["token_node_mask"][i, j, :len(nodes)] = True
        for j, (src, dst, rel, label) in enumerate(g["edges"]):
            result["edge_index"][i, j] = torch.tensor([src, dst])
            result["edge_relations"][i, j] = rel
            result["edge_targets"][i, j] = label
    return result


class Collator:
    def __init__(self, task, vocab_size, seed=17, mask_probability=0.15, graph_dropout=0.0):
        self.task, self.vocab_size = task, vocab_size
        self.generator = torch.Generator().manual_seed(seed)
        self.mask_probability, self.graph_dropout = mask_probability, graph_dropout

    def __call__(self, records):
        rows, pairs = [], []
        for r in records:
            source = len(rows)
            rows.append(r)
            if self.task == "security" and "mate" in r:
                pairs.append((source, len(rows), 0 if r["pair_relation"] == "higher" else 1))
                rows.append(r["mate"])
        b = len(rows)
        t = int(math.ceil(max(len(r["input_ids"]) for r in rows) / 8) * 8)
        batch = {"input_ids": torch.full((b, t), PAD, dtype=torch.long),
                 "attention_mask": torch.zeros(b, t, dtype=torch.bool),
                 "target_mask": torch.zeros(b, t, dtype=torch.bool),
                 "kind_ids": torch.zeros(b, t, dtype=torch.long),
                 "role_ids": torch.zeros(b, t, dtype=torch.long),
                 "language_ids": torch.zeros(b, t, dtype=torch.long),
                 "roles": torch.full((b, t, 8), -1.0)}
        for i, r in enumerate(rows):
            size = len(r["input_ids"])
            for key in ("input_ids", "target_mask", "kind_ids", "role_ids", "roles"):
                batch[key][i, :size] = torch.tensor(r[key], dtype=batch[key].dtype)
            batch["attention_mask"][i, :size] = True
            batch["language_ids"][i, :size] = r["language"]
        for key in ("label", "adequate", "weight", "cwe", "family", "teacher_probability"):
            batch[key] = torch.tensor([r[key] for r in rows], dtype=torch.float32)
        batch["pairs"] = torch.tensor(pairs, dtype=torch.long).reshape(-1, 3)
        batch["graph"] = collate_graph(rows, t)
        if self.graph_dropout and torch.rand((), generator=self.generator).item() < self.graph_dropout:
            batch["graph"] = None
        batch["meta"] = [{k: r.get(k) for k in ("id", "group_id", "repo", "offsets", "diagnostics")} for r in rows]
        if self.task == "mlm":
            original = batch["input_ids"].clone()
            eligible = (original >= 16) & batch["attention_mask"]
            selected = torch.zeros_like(eligible)
            # Span masking, with a 1..5-token distribution, independent of labels.
            for i in range(b):
                positions = eligible[i].nonzero().flatten()
                goal = max(1, round(len(positions) * self.mask_probability)) if len(positions) else 0
                attempts = 0
                while int(selected[i].sum()) < goal and attempts < t * 4:
                    at = int(positions[torch.randint(len(positions), (), generator=self.generator)])
                    width = int(torch.multinomial(torch.tensor([.45, .25, .15, .10, .05]), 1, generator=self.generator)) + 1
                    selected[i, at:min(t, at + width)] |= eligible[i, at:min(t, at + width)]
                    attempts += 1
            positions = selected.flatten().nonzero().flatten()
            if not len(positions):
                raise ValueError("Batch has no maskable code tokens")
            batch["mlm_positions"] = positions
            batch["mlm_labels"] = original.flatten().index_select(0, positions)
            coins = torch.rand(original.shape, generator=self.generator)
            masked = selected & (coins < .8)
            replaced = selected & (coins >= .8) & (coins < .9)
            random_ids = torch.randint(16, self.vocab_size, original.shape, generator=self.generator)
            batch["input_ids"][masked] = MASK
            batch["input_ids"][replaced] = random_ids[replaced]
            batch["kind_ids"][selected] = 0  # do not expose the masked lexical class
        return batch


def move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {k: v if k == "meta" else move_to_device(v, device) for k, v in value.items()}
    return value
