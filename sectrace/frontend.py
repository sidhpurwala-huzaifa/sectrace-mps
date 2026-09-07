"""Optional Clang AST/reference frontend. NOT a CFG, alias, or taint engine.

Runs the installed compiler in syntax-only mode on one supplied source unit.
Does not run target programs, tests, build scripts, or repository commands.
"""
from __future__ import annotations
import bisect
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from .util import json_records, write_jsonl


def clang_graph(code: str, language="c", compiler="clang", extra_args=None, timeout=30):
    exe = shutil.which(compiler)
    if not exe:
        raise RuntimeError("Clang not found. On macOS, install Xcode Command Line Tools (`xcode-select --install`).")
    # Flags are restricted; a compilation database can contain arbitrary flags
    # and is intentionally not executed or blindly imported by this helper.
    args = extra_args or []
    for arg in args:
        if not arg.startswith(("-I", "-D", "-U", "-std=")):
            raise ValueError("Only explicitly supplied -I/-D/-U/-std= flags are accepted")
    suffix = ".cpp" if language.lower() in ("c++", "cpp") else ".c"
    with tempfile.TemporaryDirectory(prefix="sectrace-clang-") as directory:
        path = Path(directory) / ("unit" + suffix)
        path.write_text(code, encoding="utf-8")
        proc = subprocess.run([exe, "-fsyntax-only", "-Xclang", "-ast-dump=json", *args, str(path)],
                              capture_output=True, timeout=timeout, check=False)
        try:
            ast = json.loads(proc.stdout)
        except json.JSONDecodeError:
            return None, {"frontend": "clang-ast", "parse_failed": True,
                          "message": proc.stderr.decode("utf-8", errors="replace")[:2000]}
        byte_offsets, count = [0], 0
        for ch in code:
            count += len(ch.encode("utf-8"))
            byte_offsets.append(count)
        nodes, edges, refs, declarations = [], [], [], {}
        def endpoint(p, end=False):
            if "spellingLoc" in p or "expansionLoc" in p:
                return None  # Macro mapping is ambiguous; do not invent a location.
            if "offset" not in p or p.get("includedFrom"):
                return None
            if p.get("file") and Path(p["file"]).resolve() != path.resolve():
                return None
            return int(p["offset"]) + (int(p.get("tokLen", 0)) if end else 0)
        def walk(node, parent=None):
            loc = node.get("range", {})
            lo, hi = endpoint(loc.get("begin", {})), endpoint(loc.get("end", {}), True)
            current = parent
            if lo is not None and hi is not None and 0 <= lo < hi <= count:
                a, b = bisect.bisect_left(byte_offsets, lo), bisect.bisect_left(byte_offsets, hi)
                current = len(nodes)
                nodes.append({"start": a, "end": b})
                if parent is not None:
                    edges.extend([{"src": parent, "dst": current, "type": "ast_child"},
                                  {"src": current, "dst": parent, "type": "ast_parent"}])
                if node.get("id"):
                    declarations[node["id"]] = current
                ref = node.get("referencedDecl", {}).get("id")
                if ref:
                    refs.append((ref, current))
            for child in node.get("inner", []):
                walk(child, current)
        walk(ast)
        for declaration, use in refs:
            if declaration in declarations:
                d = declarations[declaration]
                edges.extend([{"src": d, "dst": use, "type": "definition_use"},
                              {"src": use, "dst": d, "type": "use_definition"}])
        return {"nodes": nodes, "edges": edges}, {
            "frontend": "clang-ast+declaration-reference", "parse_failed": proc.returncode != 0,
            "scope": "syntactic references only; not reaching definitions or alias-sensitive dataflow",
            "message": proc.stderr.decode("utf-8", errors="replace")[:2000]}


def enrich(input_path, output, compiler="clang"):
    def rows():
        for r in json_records(input_path):
            graph, status = clang_graph(r["code"], r.get("language", "c"), compiler)
            r["graph"], r["frontend_status"] = graph, status
            # Parse diagnostics are metadata, not invented adequacy ground truth.
            if "mate" in r:
                g, s = clang_graph(r["mate"]["code"], r["mate"].get("language", "c"), compiler)
                r["mate"]["graph"], r["mate"]["frontend_status"] = g, s
            yield r
    return write_jsonl(output, rows())
