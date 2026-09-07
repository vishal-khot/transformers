# Copyright 2026 the HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

import torch
import torch.nn.functional as F
from huggingface_hub.dataclasses import strict

from ...cache_utils import Cache, DynamicCache
from ...masking_utils import create_causal_mask
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...modeling_outputs import BaseModelOutputWithPast
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
from ...processing_utils import Unpack
from ...utils import TransformersKwargs, auto_docstring, logging
from ..deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from ..deepseek_v32.configuration_deepseek_v32 import DeepseekV32Config
from ..deepseek_v32.modeling_deepseek_v32 import (
    DeepseekV32DecoderLayer,
    DeepseekV32ForCausalLM,
    DeepseekV32Indexer,
    DeepseekV32Model,
    DeepseekV32PreTrainedModel,
    DeepseekV32RotaryEmbedding,
)


logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="zai-org/GLM-5")
@strict
class GlmMoeDsaConfig(DeepseekV32Config):
    r"""
    n_group (`int`, *optional*, defaults to 1):
        Number of groups for routed experts.
    mlp_layer_types (`list`, *optional*):
        MLP type pattern for each layer (`"dense"` or `"sparse"`). Defaults to 3 dense + rest sparse.
    index_topk (`int`, *optional*, defaults to 2048):
        Number of top tokens selected by the indexer for sparse attention.
    index_head_dim (`int`, *optional*, defaults to 128):
        Head dimension for the indexer projections (DSA).
    index_n_heads (`int`, *optional*, defaults to 32):
        Number of heads for the indexer projections (DSA).
    first_k_dense_replace (`int`, *optional*, defaults to 3):
        Number of leading layers that use a dense MLP; the rest use the MoE block.
    indexer_types (`list[str]`, *optional*):
        Per-layer indexer mode (`"full"` runs the indexer, `"shared"` reuses the previous full
        layer's top-k). Defaults to the pattern derived from `index_topk_freq` /
        `index_skip_topk_offset` (or `index_topk_pattern`).
    index_num_clusters (`int`, *optional*, defaults to 512):
        Number of K-Means clusters built over the indexer key cache for IVF top-k selection.
    index_num_probes (`int`, *optional*, defaults to 64):
        Number of clusters each query probes. The scanned fraction of the key cache is roughly
        `index_num_probes / index_num_clusters`; setting the two equal recovers exact top-k.
    index_kmeans_iters (`int`, *optional*, defaults to 10):
        Number of K-Means iterations run when (re)building the index.
    index_kmeans_seed (`int`, *optional*, defaults to 0):
        Seed for K-Means centroid initialization, so clusterings are reproducible.

    ```python
    >>> from transformers import GlmMoeDsaConfig, GlmMoeDsaModel

    >>> # Initializing a GLM-MoE-DSA configuration
    >>> configuration = GlmMoeDsaConfig()

    >>> # Initializing a model from the configuration
    >>> model = GlmMoeDsaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    vocab_size: int = 154880
    hidden_size: int = 6144
    intermediate_size: int = 12288
    moe_intermediate_size: int = 2048
    num_hidden_layers: int = 78
    num_attention_heads: int = 64
    num_key_value_heads: int = 64
    n_shared_experts: int = 1
    n_routed_experts: int = 256
    routed_scaling_factor: float = 2.5
    kv_lora_rank: int = 512
    q_lora_rank: int = 2048
    qk_rope_head_dim: int = 64
    v_head_dim: int = 256
    qk_nope_head_dim: int = 192
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 8
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    index_topk: int = 2048
    index_head_dim: int = 128
    index_n_heads: int = 32
    # `"full"` runs the indexer, `"shared"` reuses the previous full layer's index mask.
    indexer_types: list[str] | None = None
    # IVF / K-Means top-k selection in the indexer (see `GlmMoeDsaIndexer`).
    index_num_clusters: int = 512
    index_num_probes: int = 64
    index_kmeans_iters: int = 10
    index_kmeans_seed: int = 0

    def __post_init__(self, **kwargs):
        # Per-layer indexer mode: a pattern (e.g. `"FSSF..."`) overrides the freq/offset schedule.
        if self.indexer_types is None:
            pattern = kwargs.get("index_topk_pattern")
            if pattern is not None:
                self.indexer_types = (
                    [{"F": "full", "S": "shared"}[c] for c in pattern] if isinstance(pattern, str) else list(pattern)
                )
            else:
                freq = max(kwargs.get("index_topk_freq", 1), 1)
                offset = kwargs.get("index_skip_topk_offset", 2)
                self.indexer_types = [
                    "full" if (max(i - offset + 1, 0) % freq) == 0 else "shared" for i in range(self.num_hidden_layers)
                ]
        super().__post_init__(**kwargs)


class GlmMoeDsaRMSNorm(DeepseekV3RMSNorm):
    pass


class GlmMoeDsaRotaryEmbedding(DeepseekV32RotaryEmbedding):
    pass


def _kmeans_cosine(
    keys: torch.Tensor,
    key_valid: torch.Tensor,
    num_clusters: int,
    num_iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Cluster indexer keys by cosine similarity, batched over the sequence dimension.

    Keys are assigned to the centroid they are most cosine-similar to, but each centroid is the
    L2-normalized mean of the **raw** keys assigned to it. Empty clusters keep their previous
    centroid instead of collapsing to the origin. Padding keys are kept out of the initial sample
    and out of the centroid means, but are still assigned, so `cluster_of_key` covers the whole
    cache; they are masked out of the scores downstream anyway.

    Args:
        keys: Indexer key cache `[B, T, D]`.
        key_valid: `True` for real tokens, `False` for padding, shape `[B, T]`.
        num_clusters: Number of clusters `C`; must be `<= T`.
        num_iters: Number of assign/update iterations.
        seed: Seed for centroid initialization, so clusterings are reproducible.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: unit-normalized FP32 centroids `[B, C, D]`, and the
            `int64` cluster index of every key, `[B, T]`.
    """
    keys = keys.float()
    head_dim = keys.shape[-1]
    keys_norm = F.normalize(keys, p=2, dim=-1)
    valid = key_valid.unsqueeze(-1).to(keys.dtype)  # [B, T, 1]
    keys_valid = keys * valid  # loop-invariant: padding must not move the centroid means

    # Seed the centroids with `num_clusters` distinct keys. Padding sorts last, so it is only drawn
    # when a sequence holds fewer real tokens than there are clusters.
    generator = torch.Generator(device=keys.device).manual_seed(seed)
    order = torch.rand(keys.shape[:2], generator=generator, device=keys.device)
    order = order.masked_fill(~key_valid, float("inf")).topk(num_clusters, dim=-1, largest=False).indices
    centroids = keys_norm.gather(1, order.unsqueeze(-1).expand(-1, -1, head_dim))

    for _ in range(num_iters):
        assignments = torch.matmul(keys_norm, centroids.transpose(-1, -2)).argmax(dim=-1)  # [B, T]
        sums = torch.zeros_like(centroids).scatter_add_(
            1, assignments.unsqueeze(-1).expand(-1, -1, head_dim), keys_valid
        )
        counts = torch.zeros_like(centroids[..., :1]).scatter_add_(1, assignments.unsqueeze(-1), valid)
        means = F.normalize(sums / counts.clamp_min(1.0), p=2, dim=-1)
        centroids = torch.where(counts > 0, means, centroids)

    assignments = torch.matmul(keys_norm, centroids.transpose(-1, -2)).argmax(dim=-1)
    return centroids, assignments


def _reorder_by_cluster(cluster_of_key: torch.Tensor, num_clusters: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Lay the keys out cluster by cluster so each cluster occupies one contiguous range.

    Args:
        cluster_of_key: Cluster index of every key, `[B, T]`.
        num_clusters: Number of clusters `C`.

    Returns:
        `tuple[torch.Tensor, torch.Tensor]`: `reordered_indices` `[B, T]`, mapping a slot in the
            cluster-contiguous layout back to its key position in the cache, and `cluster_offsets`
            `[B, C + 1]`, where cluster `c` owns slots `[offsets[c], offsets[c + 1])`.
    """
    reordered_indices = cluster_of_key.argsort(dim=-1, stable=True)
    counts = torch.zeros(
        cluster_of_key.shape[0], num_clusters, dtype=torch.long, device=cluster_of_key.device
    ).scatter_add_(1, cluster_of_key, torch.ones_like(cluster_of_key))
    cluster_offsets = F.pad(counts.cumsum(dim=-1), (1, 0))
    return reordered_indices, cluster_offsets


class GlmMoeDsaIndexer(DeepseekV32Indexer):
    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,  # Kept for BC
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        Same as [`DeepseekV32Indexer.forward`], but the indexer applies **interleaved** RoPE
        rather than the non-interleaved half-split RoPE used by DeepSeek-V3.2.

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # GLM-MoE-DSA uses interleaved RoPE in the indexer
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, T] → [B, S, T]
        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Causality needs to be taken into account when computing scores so padding tokens don't affect computation
        if attention_mask.dtype == torch.bool:
            index_scores = index_scores.masked_fill(~attention_mask, float("-inf"))
        else:
            index_scores = index_scores + attention_mask

        topk = min(self.index_topk, index_scores.shape[-1])
        return index_scores.topk(topk, dim=-1).indices.to(torch.int32)  # [B, S, topk]


class GlmMoeDsaKmeansIndexer(GlmMoeDsaIndexer):
    """
    [`GlmMoeDsaIndexer`] with IVF top-k selection instead of a scan over the whole key cache.

    K-Means over the indexer keys partitions the cache into `index_num_clusters` clusters, laid out
    so each cluster is contiguous. A query scores the centroids, probes the `index_num_probes` best
    ones, and **only the keys inside those clusters are ever scored** — the full `[B, S, H, T]` score
    matrix of the dense indexer is never built. The scoring function itself is unchanged, so this
    narrows which keys compete for the top-k and `index_num_probes == index_num_clusters` recovers
    the exact result.

    The index is rebuilt on any multi-token forward and on a cache reset; during decode the appended
    keys are assigned to their nearest existing centroid, which leaves the centroids fixed between
    rebuilds.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.num_clusters: int = config.index_num_clusters
        self.num_probes: int = config.index_num_probes
        self.kmeans_iters: int = config.index_kmeans_iters
        self.kmeans_seed: int = config.index_kmeans_seed
        # Per-sequence scratch derived from the key cache, deliberately plain attributes rather than
        # buffers so they stay out of `state_dict`.
        self.centroids: torch.Tensor | None = None
        self.reordered_indices: torch.Tensor | None = None
        self.cluster_offsets: torch.Tensor | None = None
        self.indexed_len: int = 0

    def _refresh_index(self, keys: torch.Tensor, key_valid: torch.Tensor, seq_len: int) -> None:
        """
        Bring the IVF index in sync with the indexer key cache `[B, T, D]`.

        Rebuilds from scratch on a multi-token forward, on the first call, or whenever the cache no
        longer matches what was indexed (a shorter cache means a new sequence). Otherwise the single
        appended key is assigned to its nearest existing centroid and spliced into the layout, which
        leaves the centroids themselves fixed between rebuilds.
        """
        total_len = keys.shape[1]
        num_clusters = min(self.num_clusters, total_len)
        stale = (
            self.centroids is None
            or seq_len > 1
            or total_len < self.indexed_len
            or self.centroids.shape[0] != keys.shape[0]
            or self.centroids.shape[1] != num_clusters
            or self.centroids.device != keys.device
        )
        if stale:
            self.centroids, cluster_of_key = _kmeans_cosine(
                keys, key_valid, num_clusters, self.kmeans_iters, self.kmeans_seed
            )
            self.reordered_indices, self.cluster_offsets = _reorder_by_cluster(cluster_of_key, num_clusters)
        elif total_len > self.indexed_len:
            self._append_key(keys)
        self.indexed_len = total_len

    def _append_key(self, keys: torch.Tensor) -> None:
        """Assign the newly cached key to its nearest centroid and splice it into the layout."""
        position = keys.shape[1] - 1
        new_key = F.normalize(keys[:, position:].float(), p=2, dim=-1)  # [B, 1, D]
        cluster = torch.matmul(new_key, self.centroids.transpose(-1, -2)).argmax(dim=-1)  # [B, 1]

        # The key lands at the end of its cluster's range; everything after it shifts one slot right.
        slot = self.cluster_offsets.gather(-1, cluster + 1)  # [B, 1]
        old = torch.arange(position, device=keys.device).unsqueeze(0)  # [1, T]
        shifted = torch.empty((keys.shape[0], position + 1), dtype=self.reordered_indices.dtype, device=keys.device)
        shifted.scatter_(1, old + (old >= slot).long(), self.reordered_indices)
        self.reordered_indices = shifted.scatter_(1, slot, position)

        bumped = torch.arange(self.cluster_offsets.shape[1], device=keys.device).unsqueeze(0) > cluster
        self.cluster_offsets = self.cluster_offsets + bumped.long()

    def _probe(self, q: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Pick the clusters each query searches, `[B, S, P]`.

        Cluster scoring mirrors the per-key path — every head scores the centroids, the scores are
        ReLU'd, and the same per-head weights combine them — so all heads of a query agree on one
        shared set of probed clusters. Routing is by cosine similarity, unlike the raw dot product
        that ranks the keys themselves.
        """
        num_probes = min(self.num_probes, self.centroids.shape[1])
        cluster_scores = torch.matmul(q, self.centroids.unsqueeze(1).transpose(-1, -2))  # [B, S, H, C]
        cluster_scores = F.relu(cluster_scores)

        # Combine across heads exactly as the key scores are combined: [B, S, 1, H] @ [B, S, H, C]
        probe_scores = torch.matmul(weights.unsqueeze(-2), cluster_scores).squeeze(-2)  # [B, S, C]
        return probe_scores.topk(num_probes, dim=-1).indices

    def _gather_candidates(self, probes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Expand probed clusters into the key indices they contain.

        Each probed cluster gets its own slot of `L` entries, `L` being the widest cluster this
        batch probes, so a query's candidates live in a `[P, L]` block that is simply flattened —
        clusters shorter than `L` leave their tail unfilled. Sizing by the widest cluster rather than by a running candidate total keeps
        the layout a direct image of "probe `P` clusters".

        Args:
            probes: Clusters probed by each query, `[B, S, P]`.

        Returns:
            `tuple[torch.Tensor, torch.Tensor]`: candidate key indices `[B, S, P * L]` and a bool
                mask marking the slots that hold a real candidate.
        """
        batch_size, seq_len = probes.shape[:2]
        offsets = self.cluster_offsets.unsqueeze(1).expand(batch_size, seq_len, -1)
        starts = offsets.gather(-1, probes)  # [B, S, P] first slot of each probed cluster
        sizes = offsets.gather(-1, probes + 1) - starts

        # Slot width is the widest cluster this batch actually probes, read off the device so it
        # is exact rather than an upper bound that drifts as decode grows the clusters. At least one
        # slot is kept so a query probing only empty clusters still yields a well-formed buffer.
        within = torch.arange(max(int(sizes.max()), 1), device=probes.device)  # [L]
        position = (starts.unsqueeze(-1) + within).flatten(2)  # [B, S, P * L]
        filled = (within < sizes.unsqueeze(-1)).flatten(2)

        layout = self.reordered_indices.unsqueeze(1).expand(batch_size, seq_len, -1)
        cache_len = layout.shape[-1]
        candidates = layout.gather(-1, position.clamp(max=cache_len - 1))

        # Candidates stay in cluster order. Ordering them by cache position would only change which
        # of several equally-scoring (ReLU-zeroed) keys wins a tie, and costs an O(M log M) sort.
        return candidates.clamp(max=cache_len - 1), filled

    @torch.no_grad()
    def forward(
        self,
        hidden_states: torch.Tensor,
        q_resid: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,  # Kept for BC
        past_key_values: Cache | None = None,
    ) -> torch.Tensor:
        """
        Selects the top-k tokens per query for DeepSeek Sparse Attention (DSA).

        Same as [`GlmMoeDsaIndexer.forward`], but the top-k is taken over the keys inside the
        probed clusters rather than over the whole key cache.

        Args:
            hidden_states: Input hidden states `[B, S, hidden_size]`.
            q_resid: Query residual from `q_a_layernorm(q_a_proj(x))`, shape `[B, S, q_lora_rank]`.
            position_embeddings: `(cos, sin)` from RotaryEmbedding.
            attention_mask: Causal mask, broadcastable to `[B, S, T]`.
            past_key_values: Cache object containing the indexer key cache for this layer.

        Returns:
            `torch.Tensor`: the `int32` top-k token indices of shape `[B, S, topk]`. The eager / SDPA paths
                turn these into an additive sparse mask; the `flash-mla` kernel consumes them directly.
        """
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)  # [B, S, 1, D]
        k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

        # GLM-MoE-DSA uses interleaved RoPE in the indexer
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_rot, q_pass], dim=-1)  # [B, S, H, D]
        k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)  # [B, S, D]

        if past_key_values is not None:
            k = past_key_values.update_indexer(k, self.layer_idx)

        weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
        q, k = q.float(), k.float()
        cache_len = k.shape[1]

        # A key is real if the last query — which sees the whole cache — is allowed to attend to it.
        if attention_mask.dtype == torch.bool:
            key_valid = attention_mask[:, -1]
        else:
            key_valid = attention_mask[:, -1] > torch.finfo(attention_mask.dtype).min / 2

        self._refresh_index(k, key_valid, seq_len)
        probes = self._probe(F.normalize(q, p=2, dim=-1), weights)
        candidates, filled = self._gather_candidates(probes)  # [B, S, M]

        # Score only the probed keys: [B, S, M, D] gathered, then [B, S, H, D] @ [B, S, D, M].
        candidate_keys = (
            k.unsqueeze(1)
            .expand(-1, seq_len, -1, -1)
            .gather(2, candidates.unsqueeze(-1).expand(-1, -1, -1, k.shape[-1]))
        )
        scores = torch.matmul(q, candidate_keys.transpose(-1, -2)) * self.softmax_scale
        scores = F.relu(scores)

        # Weight per head and sum across heads: [B, S, 1, H] @ [B, S, H, M] → [B, S, M]
        index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

        # Drop padding slots, padding keys, and keys the query may not look back at, mirroring the
        # causal filter the dense indexer gets from `attention_mask`.
        query_pos = torch.arange(cache_len - seq_len, cache_len, device=k.device).view(1, seq_len, 1)
        allowed = filled & (candidates <= query_pos)
        allowed &= key_valid.unsqueeze(1).expand(-1, seq_len, -1).gather(-1, candidates)
        index_scores = index_scores.masked_fill_(~allowed, float("-inf"))

        # Unfilled ranks repeat an already-selected key: `scatter` is idempotent, so a duplicate adds
        # nothing, whereas an arbitrary index would hand the query a key it never selected.
        # The contract is `min(index_topk, T)` columns. Uneven clusters can make the padded
        # candidate buffer wider than the cache, so the rank count is clamped to the contract too.
        width = min(self.index_topk, cache_len)
        top_scores, top_slots = index_scores.topk(min(width, candidates.shape[-1]), dim=-1)
        selected = candidates.gather(-1, top_slots)
        fallback = torch.where(top_scores[..., :1] > float("-inf"), selected[..., :1], query_pos)
        selected = torch.where(top_scores > float("-inf"), selected, fallback)

        if selected.shape[-1] < width:
            selected = torch.cat([selected, fallback.expand(-1, -1, width - selected.shape[-1])], dim=-1)
        return selected.to(torch.int32)  # [B, S, topk]


class GlmMoeDsaAttention(DeepseekV3Attention):
    """
    DeepSeek-V3 MLA + a DSA indexer, extended with **cross-layer top-k sharing**.

    `config.indexer_types[layer_idx]` decides whether this layer runs its own indexer (`"full"`) or
    reuses the previous full layer's top-k selection (`"shared"`).
    `next_skip_topk` signals that the *next* layer will reuse this
    layer's top-k, so it is propagated upward via `prev_topk_indices`.
    """

    def __init__(self, config: GlmMoeDsaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # Refer: https://arxiv.org/abs/2603.12201 for more details.
        self.skip_topk = config.indexer_types[layer_idx] == "shared"
        self.indexer = None if self.skip_topk else GlmMoeDsaKmeansIndexer(config, layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor,
        past_key_values: Cache | None = None,
        position_ids: torch.Tensor | None = None,
        prev_topk_indices: torch.Tensor | None = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)

        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q_states = self.q_b_proj(q_resid).view(query_shape).transpose(1, 2)
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        # Both latents are viewed as single-head, 4D tensors, as expected by `expand_kv`
        k_pass = self.kv_a_layernorm(kv_pass).view(batch_size, 1, seq_length, self.kv_lora_rank)

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        cos, sin = position_embeddings
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)

        query_states = torch.cat((q_pass, q_rot), dim=-1)

        key_states, value_states = self.expand_kv(k_pass, k_rot)

        # Sparse-attention models cache the expanded K/V, not the compressed latents. TODO (remi-or): fix this with topk
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # DSA: select this layer's top-k tokens, or reuse the previous full layer's on `"shared"` layers.
        if self.indexer is not None:
            topk_indices = self.indexer(
                hidden_states,
                q_resid,
                position_embeddings,
                attention_mask[:, 0, :, :],
                position_ids,  # Kept for BC
                past_key_values=past_key_values,
            )  # [B, S, topk]
        else:
            if prev_topk_indices is None:
                raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
            topk_indices = prev_topk_indices

        sparse_indices = None
        if self.config._attn_implementation in ("eager", "sdpa"):
            index_mask = (
                topk_indices.new_ones((batch_size, seq_length, key_states.shape[2]), dtype=torch.bool)
                .scatter(-1, topk_indices.long(), False)
                .unsqueeze(1)
            )

            if attention_mask.dtype == torch.bool:
                attention_mask = attention_mask & ~index_mask
            else:
                attention_mask = attention_mask.masked_fill(index_mask, torch.finfo(hidden_states.dtype).min)
        else:
            sparse_indices = topk_indices

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            indices=sparse_indices,  # consumed by flash_mla_with_kvcache; ignored by eager / SDPA
            **kwargs,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, topk_indices


class GlmMoeDsaDecoderLayer(DeepseekV32DecoderLayer):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        prev_topk_indices: torch.Tensor | None = None,  # MAIN DIFF with DSV3.2
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, _, topk_indices = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            prev_topk_indices=prev_topk_indices,  # MAIN DIFF with DSV3.2
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, topk_indices


class GlmMoeDsaPreTrainedModel(DeepseekV32PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"model\.layers\.78.*"]


class GlmMoeDsaModel(DeepseekV32Model):
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
                "allow_is_causal_skip": False,  # Always force creation to account for causality in the indexer
            }
            causal_mask_mapping = {"deepseek_sparse_attention": create_causal_mask(**mask_kwargs)}

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

        topk_indices = None  # MAIN DIFF with DSV3.2
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            hidden_states, topk_indices = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                prev_topk_indices=topk_indices,  # MAIN DIFF with DSV3.2
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class GlmMoeDsaForCausalLM(DeepseekV32ForCausalLM):
    _fsdp_plan = {"lm_head": "keep_full_weight"}


__all__ = [
    "GlmMoeDsaConfig",
    "GlmMoeDsaPreTrainedModel",
    "GlmMoeDsaModel",
    "GlmMoeDsaForCausalLM",
]
