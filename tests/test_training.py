import json
from pathlib import Path
import pytest
import torch
import yaml
from safetensors.torch import load_file
from sectrace.tokenization import train_tokenizer
from sectrace.data import prepare, CachedDataset
from sectrace.training import train
from sectrace.checkpoints import resolve_checkpoint
from sectrace.util import write_jsonl


def caches(root):
    tok = root / 'tokenizer.json'
    train_tokenizer([], tok, diagnostic=True)
    for offset, split in ((0, 'train'), (100, 'validation')):
        raw = root/f'{split}.jsonl'
        write_jsonl(raw, [{'id': str(i), 'group_id': f'case-{i}', 'repo': f'project-{i}',
                          'code': f'int f{i}(int i){{return i+{i};}}', 'split': split,
                          'label': i % 2} for i in range(offset, offset + 4)])
        prepare(raw, tok, root/split, 'pretrain', 64)
    return root/'train', root/'validation'


def test_exact_cpu_resume_restores_optimizer_rng_and_cursor(tmp_path):
    tr, va = caches(tmp_path)
    cfg = {'model': {'preset': 'tiny', 'dropout': .1}, 'training': {
        'stage': 'pretrain', 'seed': 8, 'max_steps': 3, 'microbatch_size': 1,
        'gradient_accumulation': 2, 'evaluate_every': 1, 'log_every': 1,
        'warmup_steps': 1, 'learning_rate': .0001, 'graph_dropout': .1,
        'keep_checkpoints': 1}}
    path = tmp_path/'cfg.yaml'
    path.write_text(yaml.safe_dump(cfg))
    train(path, tr, va, tmp_path/'uninterrupted', device='cpu', precision='fp32')
    train(path, tr, va, tmp_path/'resumed', device='cpu', precision='fp32', stop_after=1)
    train(path, tr, va, tmp_path/'resumed', resume=tmp_path/'resumed', device='cpu', precision='fp32')
    a = resolve_checkpoint(tmp_path/'uninterrupted', prefer='latest')
    b = resolve_checkpoint(tmp_path/'resumed', prefer='latest')
    left, right = load_file(str(a/'model.safetensors')), load_file(str(b/'model.safetensors'))
    for key in left:
        torch.testing.assert_close(left[key], right[key], atol=0, rtol=0, msg=key)
    state_a = torch.load(a/'trainer.pt', weights_only=True)
    state_b = torch.load(b/'trainer.pt', weights_only=True)
    assert state_a['stream'] == state_b['stream'] and state_a['seen_tokens'] == state_b['seen_tokens']


def test_cache_tampering_is_detected(tmp_path):
    tr, _ = caches(tmp_path)
    with (tr/'tokens.jsonl').open('a') as f:
        f.write('\n')
    with pytest.raises(ValueError, match='fingerprint'):
        CachedDataset(tr)
