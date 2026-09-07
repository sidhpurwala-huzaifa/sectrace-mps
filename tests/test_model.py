import copy
import pytest
import torch
from torch.nn import functional as F
from sectrace.config import ModelConfig
from sectrace.model import SecTrace, Attention, rotary, pool_nodes
from sectrace.data import encode_record, Collator
from sectrace.diagnostics import diagnostic_record
from sectrace.losses import masked_bce, security_loss


def test_exact_parameter_budget():
    with torch.device("meta"):
        m = SecTrace(ModelConfig())
    assert m.parameter_counts()["training_model"] == 99_996_650
    assert m.parameter_counts()["inference_model"] == 99_553_002


@pytest.mark.parametrize("global_layer", [False, True])
@pytest.mark.parametrize("chunk", [1, 7, 64])
def test_chunked_attention_matches_dense_outputs_and_gradients(global_layer, chunk):
    torch.manual_seed(19)
    cfg = ModelConfig.tiny()
    cfg.query_chunk, cfg.local_radius, cfg.dropout = chunk, 3, 0.
    a = Attention(cfg, global_layer).eval()
    ref = copy.deepcopy(a)
    x = torch.randn(2, 23, cfg.dim, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    valid = torch.arange(23)[None] < torch.tensor([23, 12])[:, None]
    actual = a(x, valid)
    q, k, v = ref.qkv(xx).view(2, 23, 3, cfg.heads, cfg.dim // cfg.heads).permute(2, 0, 3, 1, 4).unbind(0)
    pos = torch.arange(23)
    q, k = rotary(q, pos, cfg.rope_base), rotary(k, pos, cfg.rope_base)
    allowed = valid[:, None, None, :].expand(-1, 1, 23, -1)
    if not global_layer:
        allowed = allowed & ((pos[:, None] - pos[None]).abs() <= cfg.local_radius)[None, None]
    qvalid = valid[:, None, :, None]
    allowed = (allowed & qvalid) | (~qvalid & (pos == 0)[None, None, None, :])
    z = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
    expected = ref.out((z * qvalid).transpose(1, 2).reshape(2, 23, cfg.dim)) * valid[..., None]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    actual.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=3e-6, rtol=3e-5)
    for p, r in zip(a.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, r.grad, atol=5e-5, rtol=5e-5)


def test_all_padding_is_finite_and_zero():
    cfg = ModelConfig.tiny()
    a = Attention(cfg, False)
    x = torch.randn(2, 32, cfg.dim, requires_grad=True)
    out = a(x, torch.zeros(2, 32, dtype=torch.bool))
    assert torch.isfinite(out).all() and out.count_nonzero() == 0
    out.sum().backward()
    assert x.grad.count_nonzero() == 0


def test_eval_attention_disables_dropout():
    c = ModelConfig.tiny()
    c.dropout = .4
    a = Attention(c, True).eval()
    x, valid = torch.randn(1, 16, c.dim), torch.ones(1, 16, dtype=torch.bool)
    torch.testing.assert_close(a(x, valid), a(x, valid), atol=0, rtol=0)


def test_checkpointed_graph_backprop_matches_direct(tokenizer):
    c = ModelConfig.tiny()
    c.dropout = 0
    model = SecTrace(c).train()
    direct = copy.deepcopy(model)
    direct.cfg.checkpoint_blocks = False
    batch = Collator("security", 272)(encode_record(diagnostic_record(), tokenizer, 128))
    for m in (model, direct):
        loss, _ = security_loss(m(batch), batch)
        loss.backward()
    for (n, p), (_, q) in zip(model.named_parameters(), direct.named_parameters()):
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, atol=3e-6, rtol=3e-5, msg=n)
    assert model.adapters['1'].up.weight.grad.abs().sum() > 0
    assert model.adapters['1'].gate.grad.abs() > 0


def test_mlm_logits_only_masked_positions_and_tied_embedding(tokenizer):
    m = SecTrace(ModelConfig.tiny())
    m.set_stage("pretrain")
    b = Collator("mlm", 272)(encode_record(diagnostic_record(), tokenizer, 128, pretrain=True))
    logits = m(b, "mlm")["mlm"]
    assert logits.shape == (b['mlm_labels'].numel(), 512)
    F.cross_entropy(logits, b['mlm_labels']).backward()
    assert m.token.weight.grad is not None
    assert m.heads.risk.weight.grad is None
    assert not any('decoder.weight' in key for key in m.state_dict())
    assert (b['mlm_labels'] >= 16).all()


def test_unknown_targets_contribute_no_gradient():
    z = torch.tensor([[1., 2., 3.]], requires_grad=True)
    loss = masked_bce(z, torch.tensor([[1., -1., 0.]]), torch.ones(1))
    loss.backward()
    assert z.grad[0, 1] == 0
    assert z.grad[0, 0] != 0 and z.grad[0, 2] != 0


def test_pool_nodes_matches_mean():
    h = torch.randn(2, 9, 7, requires_grad=True)
    g = {'spans': torch.tensor([[[1, 4], [3, 7]], [[0, 2], [4, 9]]]),
         'node_mask': torch.ones(2, 2, dtype=torch.bool)}
    expected = torch.stack([torch.stack([h[b, a:z].mean(0) for a, z in g['spans'][b]]) for b in range(2)])
    torch.testing.assert_close(pool_nodes(h, g), expected, atol=1e-6, rtol=1e-5)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="Apple MPS hardware unavailable")
def test_actual_mps_forward_backward(tokenizer):
    from sectrace.data import move_to_device
    m = SecTrace(ModelConfig.tiny()).to('mps')
    b = move_to_device(Collator('security', 272)(encode_record(diagnostic_record(), tokenizer, 128)), 'mps')
    loss, _ = security_loss(m(b), b)
    loss.backward()
    assert torch.isfinite(loss).item()
