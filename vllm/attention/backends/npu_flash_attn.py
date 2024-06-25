from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch_npu

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata,
                                              AttentionMetadataPerStage)
from vllm.attention.ops.paged_attn import (PagedAttention,
                                           PagedAttentionMetadata)

SHARE_MASK_TRIL = None
SHARE_MASK_TRIL_PREFIX_CACHE =None

class FlashAttentionBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def make_metadata(*args, **kwargs) -> "FlashAttentionMetadata":
        return FlashAttentionMetadata(*args, **kwargs)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return PagedAttention.get_kv_cache_shape(num_blocks, block_size,
                                                 num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: Dict[int, int],
    ) -> None:
        PagedAttention.swap_blocks(src_kv_cache, dst_kv_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: Dict[int, List[int]],
    ) -> None:
        PagedAttention.copy_blocks(kv_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadataPerStage,
                             PagedAttentionMetadata):
    is_prompt: bool
    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_query_len: Optional[int]
    max_seq_len: Optional[int]

    subquery_start_loc: Optional[torch.Tensor]
    seq_start_loc: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]

    block_tables: Optional[torch.Tensor]
    # slot_mapping: Optional[torch.Tensor]
    slot_indices: Optional[torch.Tensor]
    # attn_mask: Optional[torch.Tensor]

    use_cuda_graph: bool


class FlashAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        sliding_window: Optional[int] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.sliding_window = sliding_window
        self.mask_type = 0
        if alibi_slopes is not None:
            self.mask_type = 2
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata[FlashAttentionMetadata],
        kv_scale: float,
    ) -> torch.Tensor:

        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            if getattr(attn_metadata, "prefill_metadata", None):
                slot_indices = attn_metadata.prefill_metadata.slot_indices
            else:
                slot_indices = attn_metadata.decode_metadata.slot_indices

            PagedAttention.write_to_paged_cache(key, value, key_cache,
                                                value_cache,
                                                slot_indices)

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                output = self._run_npu_prompt_flash_attention_forward(query, key, value, prefill_meta)
            else:
                raise NotImplemented("Prefix-enabled not support.")
        elif decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                self.num_heads,
                decode_meta.block_tables,
                decode_meta.seq_lens,
                decode_meta.max_seq_len,
                attn_metadata.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                None,
                kv_scale,
                mask_type=self.mask_type
            )
        return output

    def _run_npu_prompt_flash_attention_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata
    ) -> torch.Tensor:
        batch_size = len(attn_metadata.seq_lens)
        if getattr(attn_metadata, "attn_mask", None) is None:
            query_len = attn_metadata.seq_lens_tensor
            kv_len = torch.zeros_like(query_len).to(torch.long)
            attention_mask = gen_input_mask(batch_size, attn_metadata.max_seq_len, query_len, kv_len)
            attn_metadata.attn_mask = attention_mask
        # if self.alibi_slopes is not None and getattr(attn_metadata, "pse_shift", None):
        #     attn_metadata.pse_shift = _make_alibi_bias(
        #         self.alibi_slopes, self.num_kv_heads, batch_size,
        #         attn_metadata.max_seq_len, query.dtype)
        query = query.view(-1, attn_metadata.max_seq_len, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(-1, attn_metadata.max_seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        value = value.view(-1, attn_metadata.max_seq_len, self.num_kv_heads, self.head_size).transpose(1, 2)
        scale_fa = 1 / (self.head_size ** 0.5)
        output = torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout='BNSD',
            # pse_shift=attn_metadata.pse_shift,
            atten_mask=attn_metadata.attn_mask,
            scale_value=scale_fa,
            pre_tokens=65535,
            next_tokens=0
        )
        output = output.transpose(1, 2).reshape(batch_size, -1, self.num_heads * self.head_size)
        return output


def gen_input_mask(batch_size,
                   seq_len,
                   query_len: torch.LongTensor,
                   kv_len: torch.LongTensor):
    global SHARE_MASK_TRIL
    if SHARE_MASK_TRIL is None or SHARE_MASK_TRIL.shape[0] < seq_len:
        SHARE_MASK_TRIL = ~torch.tril(torch.ones((seq_len, seq_len),
                                                 dtype=torch.bool, device='npu'))
    range_index = torch.arange(seq_len, device=query_len.device).expand(batch_size, -1)
    select_index = range_index + kv_len.unsqueeze(1)
    attn_mask = torch.index_select(SHARE_MASK_TRIL, index=select_index.view(-1), dim=0).view(
        batch_size, seq_len, -1)
    padding_index = range_index >= query_len.unsqueeze(dim=1)
    padding_index = padding_index.unsqueeze(dim=2)
    attn_mask = attn_mask.masked_fill(padding_index, 1)
    q_l = attn_mask.shape[1]
    attn_mask = attn_mask[:, :, :q_l]
    return attn_mask.unsqueeze(dim=1)


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype
):
    bias = torch.arange(seq_len, dtype=dtype, device='npu')
    bias = bias[None, :] - bias[:, None]
    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device='npu',
        dtype=dtype
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))
    return bias
