import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from flash_attn import (
    # flash_attn_func,
    # flash_attn_kvpacked_func,
    flash_attn_qkvpacked_func,
    # flash_attn_varlen_func,
    # flash_attn_varlen_kvpacked_func,
    # flash_attn_varlen_qkvpacked_func,
    # flash_attn_with_kvcache,
)
from flash_attn.bert_padding import pad_input, unpad_input
from flash_attn.flash_attn_interface import _get_block_size
# from flash_attn.layers.rotary import apply_rotary_emb

MAX_HEADDIM_SM8x = 192


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)


def generate_random_padding_mask(max_seqlen, batch_size, device, mode="random"):
    assert mode in ["full", "random", "third"]
    if mode == "full":
        lengths = torch.full((batch_size, 1), max_seqlen, device=device, dtype=torch.int32)
    elif mode == "random":
        lengths = torch.randint(
            max(1, max_seqlen - 20), max_seqlen + 1, (batch_size, 1), device=device
        )
    elif mode == "third":
        lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, (batch_size, 1), device=device)
    padding_mask = (
        repeat(torch.arange(max_seqlen, device=device), "s -> b s", b=batch_size) < lengths
    )
    return padding_mask


def generate_qkv(
    q, k, v, query_padding_mask=None, key_padding_mask=None, kvpacked=False, qkvpacked=False
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, d)
        k: (batch_size, seqlen_k, nheads_k, d)
        v: (batch_size, seqlen_k, nheads_k, d)
        query_padding_mask: (batch_size, seqlen), bool
        key_padding_mask: (batch_size, seqlen), bool
    """
    assert not (kvpacked and qkvpacked)
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    assert k.shape == (batch_size, seqlen_k, nheads_k, d)
    assert v.shape == (batch_size, seqlen_k, nheads_k, d)

    if query_padding_mask is not None:
        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, query_padding_mask)
        output_pad_fn = lambda output_unpad: pad_input(
            output_unpad, indices_q, batch_size, seqlen_q
        )
    else:
        q_unpad = rearrange(q, "b s h d -> (b s) h d")
        cu_seqlens_q = torch.arange(
            0, (batch_size + 1) * seqlen_q, step=seqlen_q, dtype=torch.int32, device=q_unpad.device
        )
        max_seqlen_q = seqlen_q
        output_pad_fn = lambda output_unpad: rearrange(
            output_unpad, "(b s) h d -> b s h d", b=batch_size
        )

    if key_padding_mask is not None:
        k_unpad, indices_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, key_padding_mask)
        v_unpad, _, _, _ = unpad_input(v, key_padding_mask)
    else:
        k_unpad = rearrange(k, "b s h d -> (b s) h d")
        v_unpad = rearrange(v, "b s h d -> (b s) h d")
        cu_seqlens_k = torch.arange(
            0, (batch_size + 1) * seqlen_k, step=seqlen_k, dtype=torch.int32, device=k_unpad.device
        )
        max_seqlen_k = seqlen_k

    if qkvpacked:
        assert (query_padding_mask == key_padding_mask).all()
        assert nheads == nheads_k
        qkv_unpad = torch.stack([q_unpad, k_unpad, v_unpad], dim=1)
        qkv = torch.stack([q, k, v], dim=2)
        if query_padding_mask is not None:
            dqkv_pad_fn = lambda dqkv_unpad: pad_input(dqkv_unpad, indices_q, batch_size, seqlen_q)
        else:
            dqkv_pad_fn = lambda dqkv_unpad: rearrange(
                dqkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            qkv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            max_seqlen_q,
            qkv.detach().requires_grad_(),
            output_pad_fn,
            dqkv_pad_fn,
        )
    elif kvpacked:
        kv_unpad = torch.stack([k_unpad, v_unpad], dim=1)
        kv = torch.stack([k, v], dim=2)
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dkv_pad_fn = lambda dkv_unpad: pad_input(dkv_unpad, indices_k, batch_size, seqlen_k)
        else:
            dkv_pad_fn = lambda dkv_unpad: rearrange(
                dkv_unpad, "(b s) t h d -> b s t h d", b=batch_size
            )
        return (
            q_unpad.detach().requires_grad_(),
            kv_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            kv.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dkv_pad_fn,
        )
    else:
        dq_pad_fn = output_pad_fn
        if key_padding_mask is not None:
            dk_pad_fn = lambda dk_unpad: pad_input(dk_unpad, indices_k, batch_size, seqlen_k)
        else:
            dk_pad_fn = lambda dk_unpad: rearrange(dk_unpad, "(b s) h d -> b s h d", b=batch_size)
        return (
            q_unpad.detach().requires_grad_(),
            k_unpad.detach().requires_grad_(),
            v_unpad.detach().requires_grad_(),
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            q.detach().requires_grad_(),
            k.detach().requires_grad_(),
            v.detach().requires_grad_(),
            output_pad_fn,
            dq_pad_fn,
            dk_pad_fn,
        )


def construct_local_mask(
    seqlen_q,
    seqlen_k,
    window_size=(-1, -1),  # -1 means infinite window size
    query_padding_mask=None,
    key_padding_mask=None,
    device=None,
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device, dtype=torch.long), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device, dtype=torch.long)
    sk = (
        seqlen_k
        if key_padding_mask is None
        else rearrange(key_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    sq = (
        seqlen_q
        if query_padding_mask is None
        else rearrange(query_padding_mask.sum(-1), "b -> b 1 1 1")
    )
    if window_size[0] < 0:
        return col_idx > row_idx + sk - sq + window_size[1]
    else:
        sk = torch.full_like(col_idx, seqlen_k) if key_padding_mask is None else sk
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + sk - sq + window_size[1], sk),
            col_idx < row_idx + sk - sq - window_size[0],
        )


def attention_ref(
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
    alibi_slopes=None,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
        causal: whether to apply causal masking
        window_size: (int, int), left and right window size
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling k, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if alibi_slopes is not None:
        i_size = q.shape[1]
        j_size = k.shape[1]
        n_heads= q.shape[2]
        device = q.device
        context_position  = torch.arange(i_size, device=device)[:, None]
        memory_position   = torch.arange(j_size, device=device)[None, :]
        relative_position = torch.abs(memory_position - context_position)
        # [n_heads, max_token_length, max_token_length]
        relative_position = relative_position.unsqueeze(0).expand(n_heads, -1, -1)
        slopes = torch.Tensor(alibi_slopes).to(device) * 1
        alibi = slopes.unsqueeze(1).unsqueeze(1) * relative_position
        alibi = alibi.unsqueeze(0)
        # get the upper triangular part of the matrix
        #alibi = torch.tril(alibi, diagonal=1)
        print(alibi.shape)
        print(alibi[0,0])
        #scores = torch.einsum("bihd,bhij,bjhd->bhij", q , alibi ,k)
        scores = torch.einsum("bthd,bshd->bhts", q , k)
        scores = scores + alibi
    else:
        scores = torch.einsum("bthd,bshd->bhts", q , k)
    attention = scores
    
    
    # if not reorder_ops:
    #     scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    # else:
    #     scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))
    # if key_padding_mask is not None:
    #     scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, 0)
        print("========== local mask ===========")
        print(scores[0,0])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    # attention = torch.softmax(scores, dim=-1)
    # #attention = scores
    # # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    output = torch.einsum("bhts,bshd->bthd", attention, v )
    # # We want to mask here so that the attention matrix doesn't have any NaNs
    # # Otherwise we'll get NaN in dV
    # if query_padding_mask is not None:
    #     attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    # dropout_scaling = 1.0 / (1 - dropout_p)
    # # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    # if dropout_mask is not None:
    #     attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    # else:
    #     attention_drop = attention
    # output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    # if query_padding_mask is not None:
    #output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)


def attention_kvpacked_ref(
    q,
    kv,
    query_padding_mask=None,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
):
    return attention_ref(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        query_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
    )


def attention_qkvpacked_ref(
    qkv,
    key_padding_mask=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    upcast=True,
    reorder_ops=False,
    alibi_slopes=None,
):
    return attention_ref(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        key_padding_mask,
        key_padding_mask,
        dropout_p,
        dropout_mask,
        upcast=upcast,
        causal=causal,
        window_size=window_size,
        reorder_ops=reorder_ops,
        alibi_slopes=alibi_slopes
    )


def generate_sparsity_mask(seqlen, sparsity=0.3):
    repeats = seqlen // 16 // 2
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([0, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda'),
    #                     torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 1] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    # mask = torch.stack([torch.tensor([1, 0] * repeats, dtype=torch.bool, device='cuda')], dim=-1)
    nrow, ncol = seqlen // 16, seqlen // 256
    mask = torch.rand(nrow, ncol, device="cuda") < sparsity
    return mask


def attention_blocksparse_ref(qkv, blockmask, attn_mask, dropout_p, dropout_mask):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        blockmask: (seqlen / 16, seqlen / 256)
        attn_mask: (batch_size, seqlen)
        dropout_p: float
        dropout_mask: (batch_size, nheads, seqlen, seqlen)
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
        attention: softmax after dropout
    """
    q, k, v = qkv.float().unbind(dim=2)
    d = qkv.shape[-1]
    seqlen = qkv.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    blockmask = repeat(blockmask, "s_16 s_256 -> (s_16 16) (s_256 256)")
    blockmask = blockmask[:seqlen, :seqlen]
    scores.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), float("-inf"))
    attention = torch.softmax(scores, dim=-1)
    attention = attention.masked_fill(rearrange(~attn_mask, "b s -> b 1 s 1"), 0.0)
    attention = attention.masked_fill_(rearrange(~blockmask, "t s -> 1 1 t s"), 0.0)
    attention_drop = attention.masked_fill(~dropout_mask, 0.0) / (1 - dropout_p)
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v)
    output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1 1"), 0)
    return output.to(dtype=qkv.dtype), attention.to(dtype=qkv.dtype)


def convert_flash_attn_S_to_softmax(
    S,
    seqlen_q,
    seqlen_k,
    query_padding_mask,
    key_padding_mask,
    head_dim,
    is_dropout,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """FlashAttention stores the S matrix in a different way.
    Arguments:
        S: (batch_size, nheads, seqlen_q_rounded, seqlen_k_rounded)
        query_padding_mask: (batch_size, seqlen_q_rounded)
        key_padding_mask: (batch_size, seqlen_k_rounded)
    """
    if causal:
        window_size = (window_size[0], 0)
    seqlen_q_rounded, seqlen_k_rounded = S.shape[-2:]
    warps_n = 4
    blocksize_m, blocksize_n = _get_block_size(S.device, head_dim, is_dropout, causal)
    nblocks_n = (seqlen_k_rounded + blocksize_n - 1) // blocksize_n
    nblocks_m = (seqlen_q_rounded + blocksize_m - 1) // blocksize_m
    mmas_n = (blocksize_n + 16 - 1) // 16
    S_flat = rearrange(
        S,
        "b h (nblocks_m blocksize_m) (nblocks_n blocksize_n) -> b h nblocks_m nblocks_n (blocksize_m blocksize_n)",
        blocksize_m=blocksize_m,
        blocksize_n=blocksize_n,
    )
    S_converted = rearrange(
        S_flat,
        "b h nblocks_m nblocks_n (mmas_n mmas_m warps_n eight four c2 c1 c0) -> b h (nblocks_m mmas_m warps_n c1 eight) (nblocks_n mmas_n c2 four c0)",
        mmas_n=mmas_n,
        warps_n=warps_n,
        eight=8,
        c0=2,
        c1=2,
        c2=2,
        four=4,
    )

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            S.device,
        )
        local_mask = F.pad(
            local_mask,
            (0, seqlen_k_rounded - seqlen_k, 0, seqlen_q_rounded - seqlen_q),
            value=True,
        )
        S_converted.masked_fill_(local_mask, 0.0)

    # Need to zero out things not in attention_mask in case S was initialized with random values
    # and some of those values aren't overwritten.
    seqlen_q_og = (
        query_padding_mask.shape[-1] if query_padding_mask is not None else seqlen_q_rounded
    )
    if query_padding_mask is not None:
        query_padding_mask = F.pad(query_padding_mask, (0, seqlen_q_rounded - seqlen_q_og))
        S_converted = S_converted.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    seqlen_k_og = key_padding_mask.shape[-1] if key_padding_mask is not None else seqlen_k
    if key_padding_mask is not None:
        key_padding_mask = F.pad(key_padding_mask, (0, seqlen_k_rounded - seqlen_k_og))
        S_converted = S_converted.masked_fill(rearrange(~key_padding_mask, "b s -> b 1 1 s"), 0.0)
    S_converted = F.pad(S_converted, (0, 0, 0, seqlen_q_og - seqlen_q_rounded))
    S_converted = F.pad(S_converted, (0, seqlen_k_og - seqlen_k_rounded))
    return S_converted[:, :, :seqlen_q, :seqlen_k]


def normalize_flash_attn_S(
    attn_unnorm,
    q,
    k,
    v,
    query_padding_mask=None,
    key_padding_mask=None,
    is_dropout=False,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k, v: (batch_size, seqlen_k, nheads, head_dim)
        key_padding_mask: (batch_size, seqlen_q)
    Output:
        softmax_lse: (batch_size, nheads, seqlen_q)
        softmax_max: (batch_size, nheads, seqlen_q)
    """
    if causal:
        window_size = (window_size[0], 0)
    q, k, v = q.float(), k.float(), v.float()
    _, seqlen_q, _, head_dim = q.shape
    seqlen_k = k.shape[1]
    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(head_dim), k)
    if key_padding_mask is not None:
        scores.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), float("-inf"))
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            q.device,
        )
        scores.masked_fill_(local_mask, float("-inf"))
    _, block_size_n = _get_block_size(scores.device, head_dim, is_dropout, causal)
    scores_block = scores.split(block_size_n, dim=-1)
    lse_block = torch.stack([torch.logsumexp(s, dim=-1) for s in scores_block], dim=-1)
    lse = torch.logsumexp(lse_block, dim=-1)
    # lse could be -inf (i.e. all values in scores are -inf), and we want to set those to inf
    # so that when we do torch.exp(m - lse), we get 0.0 instead of NaN.
    lse[lse == float("-inf")] = float("inf")
    scores_max_block = torch.stack([torch.amax(s, dim=-1) for s in scores_block], dim=-1)
    cummax_block = torch.cummax(scores_max_block.flip(-1), dim=-1).values.flip(-1).unbind(dim=-1)
    attn_unnorm_block = attn_unnorm.split(block_size_n, dim=-1)
    attn_norm = torch.cat(
        [
            a * rearrange(torch.exp(m - lse), "b h s -> b h s 1")
            for a, m in zip(attn_unnorm_block, cummax_block)
        ],
        dim=-1,
    )
    if query_padding_mask is not None:
        attn_norm.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    return attn_norm.to(dtype=attn_unnorm.dtype)


def get_dropout_fraction(
    dropout_mask,
    query_padding_mask=None,
    key_padding_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
):
    """
    dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k), bool. True means keep, False means drop.
    query_padding_mask: (batch_size, seqlen_q)
    key_padding_mask: (batch_size, seqlen_k)
    """
    if causal:
        window_size = (window_size[0], 0)
    batch_size, nheads, seqlen_q, seqlen_k = dropout_mask.shape
    dropped = ~dropout_mask
    valid = torch.ones_like(dropout_mask)
    if query_padding_mask is not None:
        dropped.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
        valid.masked_fill_(rearrange(~query_padding_mask, "b s -> b 1 s 1"), False)
    if key_padding_mask is not None:
        dropped.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
        valid.masked_fill_(rearrange(~key_padding_mask, "b s -> b 1 1 s"), False)
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q,
            seqlen_k,
            window_size,
            query_padding_mask,
            key_padding_mask,
            dropout_mask.device,
        )
        dropped.masked_fill_(local_mask, False)
        valid.masked_fill_(local_mask, False)
    dropped_total = dropped.sum()
    return dropped.sum() / valid.sum()

def rmsenorm(x):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
@pytest.mark.parametrize("dtype", ([torch.float16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("local", [False])
# @pytest.mark.parametrize("local", [True])
@pytest.mark.parametrize("causal", [False])
# @pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [32])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 64, 96, 128])
# @pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize('seqlen', [128, 256, 384, 512, 768, 1024, 2048])
@pytest.mark.parametrize("seqlen", [ 4])
# @pytest.mark.parametrize("seqlen", [128])
@pytest.mark.parametrize("dropout_p", [0])
# @pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_qkvpacked(seqlen, d, dropout_p, causal, local, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 1
    nheads = 1
    window_size = (-1, -1) if not local else torch.randint(0, seqlen, (2,))
    qkv = torch.arange(batch_size*seqlen*3*nheads*d, device=device, dtype=dtype, requires_grad=True).reshape(batch_size,seqlen,3,nheads,d)/100
    # qkv = torch.randn(
    #     batch_size, seqlen, 3, nheads, d, device=device, dtype=dtype, requires_grad=True
    # )
    alibi_slopes = torch.ones(nheads, device=device, dtype=torch.float32)*0.9
    alibi_slopes = None
    out, lse, S_dmask = flash_attn_qkvpacked_func(
        qkv, dropout_p,softmax_scale=1, causal=causal, 
        window_size=window_size, return_attn_probs=True,
        alibi_slopes=repeat(alibi_slopes, "nh -> b nh", b=batch_size) if alibi_slopes is not None else None
    )
    if dropout_p > 0.0:
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen,
            seqlen,
            None,
            None,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
        attn_unnorm = S_dmask_converted.abs()
        attn = normalize_flash_attn_S(
            attn_unnorm,
            qkv[:, :, 0],
            qkv[:, :, 1],
            qkv[:, :, 2],
            None,
            None,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_fraction = get_dropout_fraction(
            dropout_mask, None, None, causal=causal, window_size=window_size
        ).item()
        print(f"Actual dropout fraction: {dropout_fraction}")
    else:
        dropout_mask = None

    out_ref, attn_ref = attention_qkvpacked_ref(
        qkv, None, dropout_p, dropout_mask, causal=causal, window_size=window_size,alibi_slopes=alibi_slopes
    )
    # out_pt, attn_pt = attention_qkvpacked_ref(
    #     qkv, None, dropout_p, dropout_mask, causal=causal, window_size=window_size,
    #     upcast=False, reorder_ops=True,
    # )
    # v = qkv[:, :, 2].float()
    # qk = torch.einsum('bshd,bthd->bhst', qkv[:, :, 0], qkv[:, :, 1]).float()
    # if causal:
    #     causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
    #     qk.masked_fill_(causal_mask, float('-inf'))
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # p_tmp = torch.softmax(qk / math.sqrt(d), -1)
    # p_dropped = p_tmp if dropout_mask is None else p_tmp.masked_fill(~dropout_mask, 0)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # qk_max1 = torch.max(qk[:, :, 128:, 192:], -1, keepdim=True).values
    # qk_max2 = torch.max(qk[:, :, 128:, 128:], -1, keepdim=True).values
    # qk_max3 = torch.max(qk[:, :, 128:, 64:], -1, keepdim=True).values
    # qk_max4 = torch.max(qk[:, :, 128:, :], -1, keepdim=True).values
    # o1 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 192:] - qk_max1) / math.sqrt(d)), v[:, 192:])
    # o2 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 128:] - qk_max2) / math.sqrt(d)), v[:, 128:])
    # o3 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 64:] - qk_max3) / math.sqrt(d)), v[:, 64:])
    # o4 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, :] - qk_max4) / math.sqrt(d)), v[:, :])
    print("==================================")
    # print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    # print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    # print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    # print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    print("==================================")
    # out=rmsenorm(out)
    # out_ref = rmsenorm(out_ref)
    # out_pt  = rmsenorm(out_pt)
    # print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    # print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")

    print("=============Attention: q@k==============")
    print(attn_ref[0,0])

    # print(qkv[0,:,0,0])
    # print(qkv[0,:,1,0])
    # print(qkv[0,:,2,0])
    print("==========Flash Attn Result ===============")
    print(out[0,:,0])
    print("==========Ground Attn Result ===============")
    print(out_ref[0,:,0])

    # print(lse[0,0])
    # print(S_dmask[0,0].shape)
    #print(attn_ref[0,0])
    # print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    # print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")
    if dropout_p > 0.0:
        print(f"Attention max diff: {(attn - attn_ref).abs().max().item()}")
        print(f"Attention Pytorch max diff: {(attn_pt - attn_ref).abs().max().item()}")

    # g = torch.randn_like(out)
    # # do_o = (g.float() * out.float()).sum(-1)
    # # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    # if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
    #     (dqkv,) = torch.autograd.grad(out, qkv, g)
    #     (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
    #     (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
    #     print(f"dQ max diff: {(dqkv[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
    #     print(f"dK max diff: {(dqkv[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
    #     print(f"dV max diff: {(dqkv[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
    #     print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
    #     print(f"dQ Pytorch max diff: {(dqkv_pt[:, :, 0] - dqkv_ref[:, :, 0]).abs().max().item()}")
    #     print(f"dK Pytorch max diff: {(dqkv_pt[:, :, 1] - dqkv_ref[:, :, 1]).abs().max().item()}")
    #     print(f"dV Pytorch max diff: {(dqkv_pt[:, :, 2] - dqkv_ref[:, :, 2]).abs().max().item()}")
    #     print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # # Check that FlashAttention's numerical error is at most twice the numerical error
    # # of a Pytorch implementation.
    #assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    # if dropout_p > 0.0:
    #     assert (attn - attn_ref).abs().max().item() <= 2 * (attn_pt - attn_ref).abs().max().item()
    #     assert abs(dropout_fraction - dropout_p) <= (0.01 if not local else 0.025)

    # if d <= MAX_HEADDIM_SM8x or (is_sm80 or is_sm90):
    #     assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()


test_flash_attn_qkvpacked(4,5,0,True,False,torch.float16)