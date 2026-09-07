import json
import shutil
import pytest
import numpy as np
from sectrace.importers import megavul, primevul, secvuleval
from sectrace.data import encode_record, Collator, prepare, CachedDataset
from sectrace.tokenization import lexical_fingerprint, train_tokenizer, CodeTokenizer
from sectrace.schema import validate_record, CWE
from sectrace.splitting import identities, audit_files, split_corpus
from sectrace.util import write_jsonl
from sectrace.metrics import threshold_for_fpr, binary_metrics


def record(code='int f(int i){return i+1;}', **kw):
    return {'id': 'x', 'group_id': 'case-x', 'repo': 'repo-x', 'code': code, 'label': 1, **kw}


def test_supervised_overlength_is_rejected_but_mlm_windowed(tokenizer):
    r = record('int f(){' + 'x++;' * 100 + 'return x;}')
    assert encode_record(r, tokenizer, 64) == []
    windows = encode_record(r, tokenizer, 64, pretrain=True)
    assert len(windows) > 1
    assert all(w['label'] == -1 and len(w['input_ids']) <= 64 for w in windows)


def test_unknown_cwe_is_not_negative(tokenizer):
    r = record(cwes=['CWE-125'])
    enc = encode_record(r, tokenizer, 128)[0]
    assert enc['cwe'][CWE.index('CWE-125')] == 1
    assert enc['cwe'][CWE.index('CWE-416')] == -1


def test_teacher_null_remains_unknown(tokenizer):
    r = record(teacher_probability=None)
    batch = Collator('security', 272)(encode_record(r, tokenizer, 128))
    assert batch['teacher_probability'].item() == -1


def test_unicode_character_offsets(tokenizer):
    code = 'int café=1; /* 測試 */'
    ids, offsets = tokenizer.encode(code)
    assert len(ids) == len(code.encode('utf8'))
    assert all(0 <= a < b <= len(code) for a, b in offsets)
    assert any(code[a:b] == 'é' for a, b in offsets)


def test_pair_requires_review_and_collates(tokenizer):
    r = record(mate=record('int g(){return 0;}', id='y', label=0), pair_relation='higher')
    with pytest.raises(ValueError, match='verified'):
        validate_record(r)
    r['pair_verified'] = True
    b = Collator('security', 272)(encode_record(r, tokenizer, 128))
    assert b['input_ids'].shape[0] == 2
    assert b['pairs'].tolist() == [[0, 1, 0]]


def test_importers_do_not_label_fixes_or_changed_lines_as_vulnerable():
    rows = megavul({'is_vul': True, 'func_before': 'bad()', 'func': 'fixed()', 'repo_name': 'r',
                    'commit_hash': 'abc', 'cve_id': 'CVE-2024-1111', 'cwe_ids': ['CWE-125']})
    assert rows[0]['code'] == 'bad()' and rows[0]['label'] == 1
    assert rows[1]['code'] == 'fixed()' and rows[1]['label'] is None
    raw = {'idx': 1, 'func_body': 'fixed()', 'is_vulnerable': False, 'project_url': 'https://github.com/a/b',
           'commit_id': 'abc', 'cve_list': ['CVE-2024-1111', 'CVE-2024-2222'],
           'cwe_list': ['CWE-125'], 'changed_statements': ['fixed()'], 'context': 'PRIVILEGED'}
    r = secvuleval(raw)[0]
    assert r['cwes'] == [] and 'context' not in r and 'evidence' not in r
    assert 'incident:CVE-2024-2222' in identities(r)
    assert primevul({'func': 'fixed()', 'target': 0, 'project': 'p', 'cwe': 'CWE-125'})[0]['cwes'] == []


def test_provenance_does_not_become_model_input(tokenizer):
    a = record()
    b = {**a, 'commit_message': 'FIX CVE VULNERABLE', 'context': 'SECRET LABEL', 'source': 'vulnerable'}
    aa, bb = encode_record(a, tokenizer, 128)[0], encode_record(b, tokenizer, 128)[0]
    for key in ['input_ids', 'kind_ids', 'role_ids', 'target_mask']:
        assert aa[key] == bb[key]


def test_lexical_hash_removes_comments_not_string_contents():
    assert lexical_fingerprint('int x=1;/*a*/') == lexical_fingerprint('int  x = 1;')
    assert lexical_fingerprint('puts("a b");') != lexical_fingerprint('puts("ab");')


def test_audit_catches_cross_split_incident(tmp_path):
    a, b = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    write_jsonl(a, [record(split='train')])
    write_jsonl(b, [record('int g(){return 3;}', split='test', repo='repo-y')])
    with pytest.raises(ValueError, match='leakage'):
        audit_files([a, b], group_key='incident')


def test_split_quarantines_contradictory_labels(tmp_path):
    p = tmp_path / 'raw.jsonl'
    write_jsonl(p, [record(label=0), record(label=1)])
    report = split_corpus([p], tmp_path/'split')
    assert report['counts']['conflicting_labels_quarantined'] == 2


def test_threshold_respects_tied_negative_scores():
    y, p = [0, 0, 0, 0, 1], [.7, .7, .6, .2, .9]
    t = threshold_for_fpr(y, p, .25)
    assert (np.asarray(p)[:4] >= t).mean() <= .25
    assert binary_metrics([1, 1], [.2, .8])['auroc'] is None
    assert binary_metrics([-1], [.5])['n'] == 0


def test_prepare_refuses_relabeling_partition(tmp_path):
    tok, raw = tmp_path/'tok.json', tmp_path/'raw.jsonl'
    train_tokenizer([], tok, diagnostic=True)
    write_jsonl(raw, [record(split='test')])
    with pytest.raises(ValueError, match='partition'):
        prepare(raw, tok, tmp_path/'cache', 'security', 128, purpose='train')


def test_bpe_training_optional_dependency(tmp_path):
    pytest.importorskip('tokenizers', reason='Optional public-data/tokenizer dependency not installed')
    raw, tok = tmp_path/'raw.jsonl', tmp_path/'tok.json'
    write_jsonl(raw, [record('int f(){return 1;}', split='train')])
    train_tokenizer([raw], tok, vocab_size=512)
    tokenizer = CodeTokenizer(tok)
    ids, offsets = tokenizer.encode('int café=1;')
    assert ids and max(ids) < 512 and len(ids) == len(offsets)


@pytest.mark.skipif(shutil.which('clang') is None, reason='Clang not installed')
def test_clang_graph_is_source_anchored():
    from sectrace.frontend import clang_graph
    code = 'int f(int i){int x=i+1; return x;}'
    graph, status = clang_graph(code)
    assert not status['parse_failed']
    assert all(0 <= n['start'] < n['end'] <= len(code) for n in graph['nodes'])
    assert any(e['type'] == 'definition_use' for e in graph['edges'])


def test_unicode_identifier_fingerprint_is_not_dropped():
    assert lexical_fingerprint('int café=1;') != lexical_fingerprint('int caf=1;')


def test_repository_identity_matches_cross_source_github_forms():
    from sectrace.importers import repo_identity
    assert repo_identity('Owner/Repo') == repo_identity('https://github.com/owner/repo.git')
    assert repo_identity('git@github.com:Owner/Repo.git') == 'github.com/owner/repo'
