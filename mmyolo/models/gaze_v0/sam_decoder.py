import torch
from torch import Tensor, nn
import torch.nn.functional as F

import math
from typing import Tuple, Type

import matplotlib.pyplot as plt
# from fvcore.nn import FlopCountAnalysis

from mmyolo.registry import MODELS

@MODELS.register_module()
class TwoWayTransformer(nn.Module):
    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: Type[nn.Module] = nn.GELU,
        attention_downsample_rate: int = 2,
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.

        Args:
          depth (int): number of layers in the transformer
          embedding_dim (int): the channel dimension for the input embeddings
          num_heads (int): the number of heads for multihead attention. Must
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block
          activation (nn.Module): the activation to use in the MLP block
        """
        super().__init__()
        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.layers = nn.ModuleList()

        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
            
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)
        
        # self.final_attn_token_to_image = nn.ModuleList()
        # self.norm_final_attn = nn.ModuleList()
        # for i in range(depth):
        #     self.final_attn_token_to_image.append(Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate))
        #     self.norm_final_attn.append(nn.LayerNorm(embedding_dim))
                
        # self.dropout = nn.Dropout(0.5)  # ################################드롭아웃 비율 50%

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        prompt_embedding: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # neck p5 shape 변형 (bs, c, h, w) -> (bs, h*w, c)
        bs, c, h, w = image_embedding.shape  # (32, 576, 10, 10)
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1) 
        image_pe = image_pe.flatten(2).permute(0, 2, 1)
        prompt_embedding = prompt_embedding.permute(0, 2, 1)

        queries = prompt_embedding
        keys = image_embedding

        # Apply transformer blocks and final layernorm
        for i, layer in enumerate(self.layers):
            queries, keys = layer(
                queries=queries,
                keys=keys,
                query_pe=prompt_embedding,
                key_pe=image_pe,
            )

        # Apply the final attention layer from the points to the image
        q = queries + prompt_embedding
        k = keys + image_pe
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        # queries = self.dropout(queries)  # ################################드롭아웃 적용

        queries = queries.permute(0, 2, 1).unsqueeze(-1) # (32, 576, 2, 1)
        # queries = queries.permute(0, 2, 1)
        # keys = keys.permute(0, 2, 1).view(bs, c, h, w)
        return queries 
        # return keys
        # return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

        # self.dropout = nn.Dropout(0.5)  # ################################드롭아웃 비율 50%

    def forward(
        self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        # Self attention block
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attn_out = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attn_out
        
        queries = self.norm1(queries)
        # queries = self.dropout(queries)  # ################################드롭아웃 적용

        # Cross attention block, tokens attending to image embedding
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)
        # queries = self.dropout(queries)  # ################################드롭아웃 적용

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)
        # queries = self.dropout(queries)  # ################################드롭아웃 적용

        # # MoE block
        # moe_out = self.moe(queries)
        # queries = queries + moe_out
        # queries = self.norm3(queries)

        # # MoE block (mask 이용하는 방법)
        # moe_out = self.moe(queries, num_experts_per_tok=2)
        # queries = queries + moe_out
        # queries = self.norm3(queries)

        # # CoE block
        # coe_out = self.coe(queries)
        # queries = queries + coe_out
        # queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        # keys = self.dropout(keys)  # ################################드롭아웃 적용

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

        # ### DropKey 실험
        # self.gamma = nn.Parameter(torch.ones(num_heads) * 0.1)  # 초기값은 0.1로 설정

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # ### DropKey 실험
        # q_norm = F.normalize(q, p=2, dim=-1)
        # k_norm = F.normalize(k, p=2, dim=-1)
        # attn = q_norm @ k_norm.permute(0, 1, 3, 2)

        # attn = attn / self.gamma.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) # γ는 각 헤드별로 다르게 적용. gamma = [1, 8, 1, 1]
        # attn = torch.softmax(attn, dim=-1)\
        
        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn, dim=-1)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        # Clear intermediate tensors
        del q, k, v, attn
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return out
    
class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module],
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()
        self.norm = nn.LayerNorm(mlp_dim)  ### LN 적용
        # self.dropout = nn.Dropout(0.5)  # ################################드롭아웃 비율 50%

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.act(x)
        x = self.norm(x)
        # x = self.dropout(x)
        x = self.lin2(x)
        return x