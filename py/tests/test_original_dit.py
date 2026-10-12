"""Original DiT equations and initialization, independently checked."""
from copy import deepcopy

import pytest
import torch
from torch import nn

from common.dit import DiTBlock, FlowMapAtomDiT, get_pos_embedding
from common.flow_map import TorchFlowMap
from configs.ace_a_nme import get_config


class BlockReference(nn.Module):
    """DiT adaLN-Zero equations, using native LayerNorm and MultiheadAttention.

    Reference: facebookresearch/DiT/models.py (DiTBlock); packed in-projection
    is supplied by torch rather than our Attention class. need_weights=True
    selects the explicit attention path and avoids an SDPA implementation.
    """
    def __init__(self, source):
        super().__init__()
        d = source.attn.q.in_features
        self.norm1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(d, source.attn.num_heads, dropout=0., batch_first=True)
        self.mlp = nn.Sequential(deepcopy(source.mlp_fc1), nn.GELU(approximate='tanh'),
                                 deepcopy(source.mlp_fc2))
        self.condition = nn.Sequential(nn.SiLU(), deepcopy(source.modulation))
        with torch.no_grad():
            self.attn.in_proj_weight.copy_(torch.cat([source.attn.q.weight, source.attn.k.weight, source.attn.v.weight]))
            self.attn.in_proj_bias.copy_(torch.cat([source.attn.q.bias, source.attn.k.bias, source.attn.v.bias]))
            self.attn.out_proj.load_state_dict(source.attn.out.state_dict())

    def forward(self, x, c):
        sa, ca, ga, sm, cm, gm = self.condition(c).chunk(6, dim=-1)
        a = self.norm1(x) * (1 + ca[:, None]) + sa[:, None]
        a = self.attn(a, a, a, need_weights=True)[0]
        x = x + ga[:, None] * a
        return x + gm[:, None] * self.mlp(self.norm2(x) * (1 + cm[:, None]) + sm[:, None])


def test_nonzero_block_matches_canonical_values_and_first_gradients():
    torch.manual_seed(93)
    block = DiTBlock(24, 3, 4.)
    with torch.no_grad():
        block.modulation.weight.normal_(std=.1)
    # Nonzero modulation exercises attention and MLP, not an identity block.
    reference = BlockReference(block)
    x = torch.randn(3, 7, 24, requires_grad=True)
    c = torch.randn(3, 24, requires_grad=True)
    result, expected = block(x, c), reference(x, c)
    torch.testing.assert_close(result, expected, atol=4e-7, rtol=2e-6)
    actual_grad = torch.autograd.grad(result.square().mean(),
                                    (x, c, block.attn.q.weight, block.mlp_fc1.weight))
    reference_grad = torch.autograd.grad(expected.square().mean(),
                                        (x, c, reference.attn.in_proj_weight, reference.mlp[0].weight))
    reference_grad = (*reference_grad[:2], reference_grad[2][:24], reference_grad[3])
    for actual, wanted in zip(actual_grad, reference_grad):
        torch.testing.assert_close(actual, wanted, atol=1e-7, rtol=3e-5)


@pytest.mark.parametrize('density', [False, True])
def test_original_initialization_and_atom_position_features(density):
    torch.manual_seed(31)
    net = FlowMapAtomDiT(22, 3, predict_divergence=density,
                         num_shared_layers=4, num_div_layers=2, com_aware=True)
    for name, parameter in net.named_parameters():
        if 'modulation' in name or 'final_linear' in name:
            assert torch.count_nonzero(parameter) == 0, name
        elif name.endswith('weight'):
            bound = parameter.shape[1] ** -.5
            assert parameter.abs().max() <= bound, name
            assert parameter.std().item() == pytest.approx(bound / 3**.5, rel=.1), name
    assert torch.count_nonzero(net.emb_t.linear1.bias) > 0
    assert net.x_embedder.in_features == 6
    positions = torch.arange(22, dtype=torch.float32)
    phase = positions[:, None] * torch.pi / 2048 ** (torch.arange(96).float() / 96)
    expected = torch.cat((phase.sin(), phase.cos()), dim=-1)[None]
    torch.testing.assert_close(net.pos_emb, expected, atol=0, rtol=0)
    x, s, gap = torch.randn(2, 22, 3), torch.rand(2, 1, 1), torch.rand(2, 1, 1)
    velocity, div = net(x, s, gap)
    assert velocity.count_nonzero() == 0
    assert (div is not None) is density
    if density:
        assert div.count_nonzero() == 0
    assert sum(p.numel() for p in net.parameters()) == (5581254 if density else 4173699)


def test_two_independent_source_gap_embeddings_and_all_blocks_receive_the_sum():
    cfg = get_config(0)
    cfg.network.predict_divergence = False
    net = TorchFlowMap(cfg.network)
    s, t, x = torch.tensor([.2, .4]), torch.tensor([.7, .3]), torch.randn(2, 22, 3)
    seen, conditions = {}, []
    handles = [net.dit.emb_s.register_forward_pre_hook(lambda m, a: seen.update(source=a[0].clone())),
               net.dit.emb_t.register_forward_pre_hook(lambda m, a: seen.update(gap=a[0].clone()))]
    handles += [block.register_forward_pre_hook(lambda m, a: conditions.append(a[1].clone()))
                for block in net.dit.blocks]
    try:
        net.calc_phi(s, t, x)
        torch.testing.assert_close(seen['source'], s, atol=0, rtol=0)
        torch.testing.assert_close(seen['gap'], t-s, atol=0, rtol=0)
        expected = net.dit.emb_s(s) + net.dit.emb_t(t-s)
        assert len(conditions) == 6
        for condition in conditions:
            torch.testing.assert_close(condition, expected, atol=0, rtol=0)
    finally:
        for handle in handles:
            handle.remove()
    assert net.dit.emb_s.linear1.weight is not net.dit.emb_t.linear1.weight
