import torch
import torch.nn as nn
import torch.nn.functional as F


# class MultiHeadAttention(nn.Module):

#     def __init__(self, d_model, n_head):
#         super(MultiHeadAttention, self).__init__()
#         self.multihead_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_head)

#     def forward(self, q, k, v):

#         B, C, H, W = q.shape
#         cut = q

#         # B, C, H, W -> B, H*W, C
#         q = q.view(B, -1, C)
#         k = k.view(B, -1, C)
#         v = v.view(B, -1, C)
    
#         attn, attn_weights = self.multihead_attention(q, k, v)

#         attn = attn.view(B, C, H, W)
#         attn = attn + cut
        
#         return attn

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head=8, use_pos_embed=False):
        super(MultiHeadAttention, self).__init__()

        dim_out = d_model*2

        self.n_heads = n_head
        # self.dim_head = d_model // n_head
        self.dim_head = dim_out // n_head
        self.use_pos_embed = use_pos_embed

        if use_pos_embed:
            self.conv = nn.Conv2d(d_model*2, d_model, kernel_size=1)
  
        # Linear projections

        self.q_proj = nn.Linear(d_model, dim_out)
        self.k_proj = nn.Linear(d_model, dim_out)
        self.v_proj = nn.Linear(d_model, dim_out)
        self.out_proj = nn.Linear(dim_out, d_model)

    def forward(self, q, kv, pos_embed=None):
        """
        q: (B, Nq, dim_q) - query features = face_position_map
        kv: (B, Nk, dim_kv) - key/value features = x
        pos_embed: (B, Nk, d_pos) - optional positional embedding to concat to kv
        """
        if self.use_pos_embed and pos_embed is not None:
            # pos_embed 배치 차원 맞추기
            pos_embed = pos_embed.unsqueeze(0).repeat(q.shape[0], 1, 1, 1)

            q = torch.cat([q, pos_embed], dim=1)
            q = self.conv(q)

        B, C, H, W = q.shape   # q = face_position_map + pos_embed
        Nq = Nk = H*W

        q = q.view(B, Nq, -1)    # B x HW x E
        kv = kv.view(B, Nk, -1)  # B x HW x E

        # Linear projections
        Q = self.q_proj(q).view(B, Nq, self.n_heads, self.dim_head).transpose(1, 2)  # (B, heads, Nq, dim_head)
        K = self.k_proj(kv).view(B, Nk, self.n_heads, self.dim_head).transpose(1, 2) # (B, heads, Nk, dim_head)
        V = self.v_proj(kv).view(B, Nk, self.n_heads, self.dim_head).transpose(1, 2) # (B, heads, Nk, dim_head)

        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) / (self.dim_head ** 0.5)  # (B, heads, Nq, Nk)
        attn_probs = F.softmax(attn_scores, dim=-1)

        out = attn_probs @ V  # (B, heads, Nq, dim_head)
        out = out.transpose(1, 2).reshape(B, Nq, self.n_heads * self.dim_head)  # (B, Nq, dim_out)
        out = self.out_proj(out).permute(0, 2, 1)  # (B, d_model, Nq)

        return out.view(B, C, H, W)