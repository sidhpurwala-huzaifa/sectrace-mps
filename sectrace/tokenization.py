from __future__ import annotations
import bisect
import json
from pathlib import Path
import re
from .util import atomic_json, json_records

SPECIAL = ["<pad>", "<mask>", "<cls>", "<sep>", "<unk>"] + [f"<reserved_{i}>" for i in range(5, 16)]
PAD, MASK, CLS, SEP = 0, 1, 2, 3
# Preserves literal contents. This is lexical tokenization, not C++ parsing.
LEX = re.compile(r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|[^\W\d]\w*|(?:0[xX][0-9a-fA-F]+|\d+(?:\.\d+)?)|[^\w\s]', re.UNICODE)
KEYWORDS = set("if else for while do switch case return break continue sizeof struct class typedef unsigned signed int long short char void bool const volatile static auto delete new free nullptr null using namespace template true false".split())


def lexical_fingerprint(code: str) -> str:
    """Whitespace/comment-insensitive exact lexical fingerprint; not clone detection."""
    from .util import digest
    parts = [m.group() for m in LEX.finditer(code)
             if not m.group().startswith(("//", "/*"))]
    # Length framing avoids accidentally merging adjacent tokens/literals.
    return digest("".join(f"{len(s)}:{s}" for s in parts))


class CodeTokenizer:
    def __init__(self, path: str | Path):
        self.path = str(path)
        raw = json.loads(Path(path).read_text())
        self.byte = raw.get("type") == "byte-diagnostic"
        if self.byte:
            self.vocab_size = self.random_vocab_size = 272
            self.backend = None
        else:
            try:
                from tokenizers import Tokenizer
            except ImportError as e:
                raise RuntimeError("Install optional data dependencies: pip install -e '.[data]'") from e
            self.backend = Tokenizer.from_file(str(path))
            self.vocab_size = self.backend.get_vocab_size()
            meta = Path(str(path) + ".meta.json")
            self.random_vocab_size = json.loads(meta.read_text()).get("effective_vocab_size", self.vocab_size) if meta.exists() else self.vocab_size
            for i, token in enumerate(SPECIAL):
                if self.backend.token_to_id(token) != i:
                    raise ValueError("Tokenizer has incompatible special token IDs")

    def encode(self, text: str):
        if not self.byte:
            e = self.backend.encode(text, add_special_tokens=False)
            return e.ids, e.offsets
        ids, offsets = [], []
        for i, char in enumerate(text):
            for b in char.encode("utf-8"):
                ids.append(16 + b)
                offsets.append((i, i + 1))
        return ids, offsets


def train_tokenizer(inputs: list[str], output: str, vocab_size=32768, diagnostic=False):
    output = str(output)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if diagnostic:
        atomic_json(output, {"type": "byte-diagnostic", "vocab_size": 272,
                             "warning": "Offline pipeline diagnostics only; not the research tokenizer."})
        return
    from tokenizers import Tokenizer, Regex, models, trainers, pre_tokenizers, decoders
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[^\w\s]|\s+"), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  special_tokens=SPECIAL, show_progress=True)
    def texts():
        for path in inputs:
            for r in json_records(path):
                if r.get("split") not in (None, "train"):
                    raise ValueError("Tokenizer training attempted to use a non-training split")
                yield r["code"]
    tokenizer.train_from_iterator(texts(), trainer=trainer)
    effective = tokenizer.get_vocab_size()
    if effective > vocab_size:
        raise ValueError("Vocabulary budget too small")
    # Tiny corpora may not contain enough merges. Unused slots preserve the
    # architecture size but are never used as random MLM replacements.
    tokenizer.add_special_tokens([f"<unused_{i}>" for i in range(vocab_size - effective)])
    tokenizer.save(output)
    atomic_json(output + ".meta.json", {"effective_vocab_size": effective, "embedding_vocab_size": vocab_size})


def token_kinds(code: str, offsets: list[tuple[int, int]]) -> list[int]:
    spans = []
    for m in LEX.finditer(code):
        s = m.group()
        kind = (5 if s.startswith(("//", "/*")) else 4 if s.startswith(('"', "'")) else
                2 if s in KEYWORDS else 3 if s[0].isdigit() else
                1 if s[0].isalpha() or s[0] == "_" else 6)
        spans.append((m.start(), m.end(), kind))
    starts = [s[0] for s in spans]
    result = []
    for a, b in offsets:
        j = bisect.bisect_right(starts, a) - 1
        result.append(spans[j][2] if j >= 0 and a < spans[j][1] and b > spans[j][0] else 0)
    return result
