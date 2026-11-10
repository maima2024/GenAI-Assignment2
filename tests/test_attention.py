import pytest
torch = pytest.importorskip("torch")
from trialmatch.attention import reference_attention, tiled_attention


@pytest.mark.parametrize("Lq,Lk,blk,causal", [(7, 7, 3, True), (1, 33, 8, True), (5, 37, 4, True), (16, 16, 16, False), (33, 33, 8, False)])
def test_tiled_matches_reference(Lq, Lk, blk, causal):
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2, 3, Lq, 16, generator=g, dtype=torch.float64)
    k = torch.randn(2, 3, Lk, 16, generator=g, dtype=torch.float64)
    v = torch.randn(2, 3, Lk, 16, generator=g, dtype=torch.float64)
    a = reference_attention(q, k, v, causal)
    b = tiled_attention(q, k, v, causal, block_q=blk, block_k=blk)
    assert (a - b).abs().max() < 1e-10
