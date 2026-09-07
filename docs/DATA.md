# Data contracts, imports and curation

## Import support matrix

| Source | Entry point | What is and is not inferred |
|---|---|---|
| The Stack Smol | `import-hf --format code` | Requires actual content, repository identity, approved license metadata and C/C++ language |
| Authorized hydrated Stack v2 | `import --format code` | Requires content; identifiers alone rejected; normalize original metadata as described below |
| PrimeVul original JSONL | `import --format primevul` | Uses `func` and `target`; original binary label, not independently revalidated; no line labels inferred |
| MegaVul flattened JSON/JSONL | `import --format megavul` | For vulnerable rows, `func_before` is positive; `func` is post-fix and left unknown pending review |
| SecVulEval | `import --format secvuleval` or `import-hf --format secvuleval` | Uses `func_body`, `is_vulnerable`, `project_url`, `commit_id`, `cve_list`, `cwe_list`; omits generated context and patch-derived evidence claims |
| CVEfixes SQLite | `import-sqlite` | Joins method/file/fix records read-only; exports unknown-label patch candidates, not automatic positives |
| MoreFixes PostgreSQL dump | External authorized SELECT/export → `import --format canonical` | No native PostgreSQL importer; do not pass a PostgreSQL dump to SQLite |
| ARVO | Curated reproductions → canonical records | No automatic building/execution of vulnerable projects; root cause and fixed-version labels require review |
| D2A | Curated source/trace mapping → canonical records | Static-analysis outputs remain weak supervision; no automatic conversion of traces to ground truth |
| Juliet/SARD | Curated, normalized fixtures → canonical records | No automatic benchmark-label parsing; remove label-revealing names/comments consistently and hold out template families |
| DiverseVul/other datasets | Explicit schema conversion → canonical records | Not a native adapter in this release; global overlap analysis required |

Importer support is source-code/schema support, not a claim that every live remote release was downloaded and exercised in the build environment. Public access can still require upstream acknowledgement, credentials or an agreement. Do not bypass those controls.

### Additional commands

```bash
python -m sectrace import --format megavul \
  --input data/raw/megavul_simple.json --output data/canonical/megavul.jsonl

python -m sectrace import-hf --dataset arag0rn/SecVulEval \
  --format secvuleval --limit 20000 --acknowledge-terms \
  --output data/canonical/secvuleval.jsonl

python -m sectrace import-sqlite --database data/raw/CVEfixes.db \
  --output data/canonical/cvefixes-candidates.jsonl
```

CVEfixes schema variants can use `--query-file query.sql`. The query must start with SELECT and expose `code`, `repo`, and `incident`; optional fields are `id`, `language`, `before_change`, and `commit_hash`. The database is opened read-only with query-only enforcement. Review the official schema and underlying license before constructing a custom export.

### Hydrated code records

The generic local code adapter accepts `code` or `content`; repository identity in `repo`, `repository_name`, `repo_name` or `repo_url`; language in `lang` or `language`; license in `license`, `licenses` or `detected_licenses`; and path in `path`. For Stack v2, preserve the original metadata externally; the local adapter retains source revision/content identifiers when present. Keep the upstream revision, source revision, file hash, permission basis and removal status in your provenance manifest. Local import is not an automatic licensing gate.

## Canonical JSONL

One complete analysis unit per line:

```json
{
  "id": "reviewed-case-0001:before",
  "group_id": "reviewed-case-0001",
  "repo": "github.com/example/project",
  "incident": "CVE-2024-0001",
  "incidents": ["CVE-2024-0001"],
  "split": "train",
  "code": "int f(int i){int a[4]={0};return a[i];}",
  "label": 1,
  "label_scope": "in-scope out-of-bounds read with untrusted index",
  "language": "c",
  "weight": 1.0,
  "cwes": ["CWE-125"],
  "known_cwes": ["CWE-125"],
  "confidence": "example only; not a real adjudicated incident"
}
```

The example's ID/CVE are illustrative, not a claim about a real vulnerability. Prefer the executable generated example in `examples/reviewed-schema.json`, whose offsets are computed from its code. Omit `target_span` to default to the full code; do not copy numerical offsets from a different string.

- `code` is the actual inference-visible source/context. Relevant declarations/callers may be included, but must match the source revision. `target_span` identifies the region being judged.
- Offsets are Unicode **character** indices, end-exclusive, not byte offsets or line numbers. The Clang helper explicitly maps compiler UTF-8 byte offsets back to characters.
- `label`: `1`, `0`, or `null`. Null means unassessed, not safe.
- `cwes`: positively annotated output slots. `known_cwes`: assessed slots, negative unless also listed in positives. Set `cwe_complete` only for genuinely exhaustive annotation over all output slots.
- Public CVE-associated CWE metadata on fixed/negative versions is **not** a positive CWE label. The supplied PrimeVul/SecVulEval importers avoid this particular error.
- `family_labels`: explicit slot-to-binary mappings, e.g. `{"0": 1}`. The 12 slots are experimental; use one documented mapping consistently across the entire project.
- `adequate`: whether this supplied context supports the requested security judgment; null means not annotated.
- `evidence`: source spans plus explicitly assessed role labels. Unlisted tokens/roles remain unknown.
- `teacher_probability`: optional finite 0–1 soft label, never an automatic gold label. Enable the teacher loss only deliberately.
- `weight`: positive finite relative confidence weight. Weighting does not repair a wrong annotation.
- `source`, advisories, commit messages, fix IDs, generated explanations and provenance remain metadata; importers do not concatenate them to model input.

### Evidence and graph examples

```json
{
  "evidence": [
    {"start": 10, "end": 15, "roles": {"guard": 1, "release": 0}}
  ],
  "graph": {
    "nodes": [{"start": 0, "end": 9}, {"start": 10, "end": 15}],
    "edges": [{"src": 0, "dst": 1, "type": "cfg_next", "evidence": null}]
  }
}
```

Replace spans with valid offsets in the actual record. `evidence` on a graph edge labels whether it supports the finding; it does **not** label whether the program edge exists. An unannotated relationship remains unknown. Source extraction may supply candidate relations without claiming security truth. The graph helper never inserts a vulnerability label.

### Verified pairs

Attach a second canonical unit in `mate`. Use `pair_relation: "higher"` only when adjudication supports higher *in-scope risk* in the first unit; use `"invariant"` for a validated semantics-preserving variant. `pair_verified: true` is mandatory. Group originals, fixes, backports, and all derived variants together. A post-fix version may still contain unrelated flaws, so a binary zero is not automatic.

Nested mates are rejected. The collator expands a pair to two separately encoded sequences. Therefore a microbatch size of one paired record can require memory for two sequences.

## Recommended dataset process

Start with a bounded pre-training corpus and PrimeVul labels. Establish the model/data/backend behavior before scaling. Reconstruct a higher-confidence set of real vulnerability cases using reviewed CVEfixes/MoreFixes candidates and public reproducible cases. Keep synthetic material supplementary and report its share.

Hard negatives must have a defined reviewed scope: a valid guard, a different allocation rather than an alias, or a relevant caller contract can change the judgment. “No CVE found” is not a strong negative label. A changed patch line is not automatically the bug location. A sanitizer-triggered defect does not automatically prove a particular security impact.

For alignment, use a separate development/mining pool to collect model errors. Record reviewer decisions, assumptions, evidence, and disagreement. Include complete-context negatives and genuinely decisive context-removal examples. Do not label every removed header/callee as “insufficient context.” Semantic invariance transformations require validation and should affect both positive and negative examples.

The code provides one-cache training with confidence weights, not an automatic 70/20/10 multi-source scheduler or active-labeling service. Construct and document your sampling mixture before caching; keep original unique-case counts distinct from repeated presentations.

## Provenance, exclusions and scale

Global repository/group/exact-lexical partitioning is implemented. GitHub `owner/repo` shorthand is normalized to match full GitHub URLs; provide alias maps for bare project names, forks, mirrors and other repository identities. Near-clone detection, fork discovery, whole-file versus function overlap, temporal information cutoffs and license/secret scanning are not comprehensive automated services here. Complete those controls before a serious benchmark claim.

Caches retain hashes of input/tokenizer/data bytes. Cache opens recompute the data hash to detect tampering; large caches incur a sequential verification read. JSONL caches are readable and convenient for a Mac pilot but can be large. Measure bytes per token and available storage; a 10B presentation budget is not a requirement to cache 10B unique tokens. This release is not a distributed preprocessing system.

Build/run vulnerable software only in an appropriately isolated environment without credentials, production access or unnecessary networking. This package does not execute untrusted dataset tests or build scripts.
