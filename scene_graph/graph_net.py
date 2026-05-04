import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
from torch_scatter import scatter_add
import math

# ---------- Networks ----------

class NodeEncoder(nn.Module):
    def __init__(self, geo_dim=19, clip_dim=768, out_dim=512, dropout=0.1):
        super().__init__()
        self.geo_mlp = nn.Sequential(
            nn.Linear(geo_dim,  64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64,  64), nn.ReLU()
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(clip_dim, clip_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(clip_dim, clip_dim), nn.ReLU()
        )
        self.fuse = nn.Sequential(
            nn.Linear(64 + clip_dim, out_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, geo_feat, clip_feat):
        #clip_feat = clip_feat.detach()  # freeze CLIP
        clip_feat = self.txt_proj(clip_feat)  # (N, 768)
        geo_feat = self.geo_mlp(geo_feat)  # (N, 768)
        node_feat = self.fuse(torch.cat([clip_feat, geo_feat], dim=-1))
        return node_feat

class EdgeEncoder(nn.Module):
    def __init__(self, geo_dim=19, node_dim=768, out_dim=512, dropout=0.1):
        super().__init__()

        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )
        self.txt_proj = nn.Sequential(
            nn.Linear(node_dim*2, out_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )
        self.fuse = nn.Sequential(
            nn.Linear(out_dim + 64, out_dim),
            nn.Sigmoid()
        )

    def get_edge_node_features(self, node_feat, edge_index):
        src, dst = edge_index[0], edge_index[1]
        edge_node_feat = torch.cat([node_feat[src], node_feat[dst]], dim=-1)  # (E, node_dim*2)
        return edge_node_feat

    def forward(self, geo_feat, node_feat, edge_index):

        edge_node_feat = self.get_edge_node_features(node_feat, edge_index)  # (E, node_dim*2)

        # geo_feat: (E, geo_dim), txt_feat: (E, 512)
        geo_emb = self.geo_proj(geo_feat)
        text_emb = self.txt_proj(edge_node_feat)
        fused_emb = self.fuse(torch.cat([text_emb, geo_emb], dim=-1))
        return F.normalize(fused_emb, dim=-1), text_emb


class GatedEdgeEncoder(nn.Module):
    """
    Gated edge encoder that fuses node context, geometric features,
    and optional prior (noisy) edge features.
    """
    def __init__(self, geo_dim=19, node_dim=768, out_dim=512, dropout=0.1):
        super().__init__()

        # geometric encoding
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64)
        )

        # node-pair (text/semantic) encoding
        self.txt_proj = nn.Sequential(
            nn.Linear(node_dim * 2, out_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        # fuse geometry + text context into a candidate edge embedding
        self.fuse = nn.Sequential(
            nn.Linear(out_dim + 64, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim)
        )

        # gating network: decides how much to trust prior vs. prediction
        self.gate_net = nn.Sequential(
            nn.Linear(out_dim * 2 + 1, out_dim), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
            nn.Sigmoid()
        )

        # learned token for missing priors
        self.missing_prior = nn.Parameter(torch.randn(out_dim))

    # ----------------------------------------------------------------------

    def get_edge_node_features(self, node_feat, edge_index):
        src, dst = edge_index
        return torch.cat([node_feat[src], node_feat[dst]], dim=-1)

    # ----------------------------------------------------------------------

    def forward(self, geo_feat, node_feat, edge_index, prior_feat=None, prior_mask=None):
        """
        geo_feat:   (E, geo_dim)
        node_feat:  (N, node_dim)
        edge_index: (2, E)
        prior_feat: (E, out_dim) or None
        prior_mask: (E, 1) float {0,1}, 1 = has prior
        """
        E = geo_feat.size(0)
        device = geo_feat.device

        # --- build prior features ---
        if prior_feat is None:
            prior_feat = torch.zeros(E, self.missing_prior.numel(), device=device)
            prior_mask = torch.zeros(E, 1, device=device)
        elif prior_mask is None:
            prior_mask = (prior_feat.norm(dim=-1, keepdim=True) > 0).float()

        # replace missing priors with learned token
        miss_token = F.normalize(self.missing_prior, dim=0).expand(E, -1)
        prior_emb = torch.where(prior_mask.bool(), prior_feat, miss_token)

        # --- predict new edge embedding from node + geo ---
        edge_node_feat = self.get_edge_node_features(node_feat, edge_index)
        geo_emb = self.geo_proj(geo_feat)
        text_emb = self.txt_proj(edge_node_feat)
        pred_emb = self.fuse(torch.cat([text_emb, geo_emb], dim=-1))  # (E, out_dim)

        # --- compute gating weight ---
        gate_in = torch.cat([pred_emb, prior_emb, prior_mask], dim=-1)
        gate = self.gate_net(gate_in)  # (E, 1) in [0,1]
        gate = gate * prior_mask  # ignore gate when no prior

        # --- fuse prediction and prior ---
        fused = F.normalize(gate * prior_emb + (1 - gate) * pred_emb, dim=-1)

        return fused, gate

# -------------------------------------------------
# 1)  Prior-aware edge encoder  (new)
# -------------------------------------------------
class PriorAwareEdgeEncoder(nn.Module):
    """
    Preserves known semantic priors (edge_init) while learning to
    refine/denoise them and predict new ones from geometry.
    """
    def __init__(self, geo_dim=19, txt_dim=512, out_dim=512,
                 hidden=512, dropout=0.1, use_trust_gate=True):
        super().__init__()
        self.use_trust_gate = use_trust_gate

        # geometric projection for unknown edges
        self.geo_proj = nn.Sequential(
            nn.Linear(geo_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim)
        )

        # light denoiser for priors
        self.refine = nn.Sequential(
            nn.Linear(out_dim + geo_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

        # optional soft trust weighting
        if use_trust_gate:
            self.trust = nn.Sequential(
                nn.Linear(out_dim + geo_dim, hidden // 2),
                nn.ReLU(),
                nn.Linear(hidden // 2, 1),
                nn.Sigmoid()
            )

    def forward(self, edge_geo, edge_init, known_mask=None):
        geo_feat = F.normalize(self.geo_proj(edge_geo), dim=-1)

        if known_mask is None:
            # no priors at all
            return geo_feat

        known_mask = known_mask.unsqueeze(-1).float()

        # slight refinement of priors
        refine_in = torch.cat([edge_geo, edge_init], dim=-1)
        prior_refine = F.normalize(
            edge_init + 0.1 * self.refine(refine_in),
            dim=-1
        )

        if self.use_trust_gate:
            trust = self.trust(refine_in)         # (E,1) in [0,1]
            fused_known = trust * prior_refine + (1 - trust) * geo_feat
        else:
            fused_known = prior_refine

        fused = geo_feat * (1 - known_mask) + fused_known * known_mask
        return F.normalize(fused, dim=-1)

# -------------------------------------------------
# 2)  Direction-aware bidirectional residual block

class MultiHeadAttentionLayerWithEdge(nn.Module):
    """
    Edge-aware attention *and* edge update.
    - Edges bias attention logits (as before).
    - Edges get updated via a node-conditioned residual MLP:
        e' = e + gate(e) * MLP([x_src, x_dst, e])
      then L2-normalized and returned as [E, H, D].
    """
    def __init__(self, in_dim, out_dim, num_heads,
                 use_bias=False, attn_dropout=0.0, edge_dropout=0.0):
        super().__init__()
        self.out_dim = out_dim                  # per-head dim D
        self.num_heads = num_heads              # heads H
        self.attn_dropout = attn_dropout
        self.edge_dropout = edge_dropout

        # Node projections (attention space: in_dim == H*D)
        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        # Edge projection used inside attention logits
        self.proj_e = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)

        # Edge update: node-conditioned residual (works in the same in_dim space)
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_dim * 3, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim)
        )
        self.edge_gate = nn.Linear(in_dim, in_dim)  # produces gate for residual
        self.edge_ln   = nn.LayerNorm(in_dim)       # stabilize edge stream

        self.eps = 1e-6
        self.scale = 1.0 / math.sqrt(out_dim)

    def forward(self, x, edge_index, edge_attr):
        """
        x:          [N, in_dim]          node features (attention space, e.g., 512)
        edge_index: [2, E]               (src -> dst)
        edge_attr:  [E, in_dim]          edge features (same in_dim as nodes)
        Returns:
            h_out: [N, H, D]
            e_out: [E, H, D]
        """
        N = x.size(0)
        E = edge_index.size(1)
        H, D = self.num_heads, self.out_dim

        # ----- Projections -----
        Q_h = self.Q(x).view(N, H, D)   # dst uses Q
        K_h = self.K(x).view(N, H, D)   # src uses K
        V_h = self.V(x).view(N, H, D)   # src uses V
        proj_e = self.proj_e(edge_attr).view(E, H, D)  # edge term for logits

        src, dst = edge_index[0], edge_index[1]
        Q_dst = Q_h[dst]          # [E, H, D]
        K_src = K_h[src]          # [E, H, D]
        V_src = V_h[src]          # [E, H, D]

        # ----- Edge-aware attention logits -----
        logits = (Q_dst * K_src).sum(-1) * self.scale                     # [E, H]
        logits = logits + (proj_e * K_src).sum(-1) * self.scale           # [E, H]

        # Softmax per-dst, per-head
        attn = softmax(logits, dst)                                       # [E, H]
        if self.training and self.attn_dropout > 0:
            attn = F.dropout(attn, p=self.attn_dropout)

        # ----- Node aggregation -----
        m  = attn.unsqueeze(-1) * V_src                                   # [E, H, D]
        wV = scatter_add(m, dst, dim=0, dim_size=N)                       # [N, H, D]
        z  = scatter_add(attn, dst, dim=0, dim_size=N)                    # [N, H]
        h_out = wV / (z.unsqueeze(-1) + self.eps)                         # [N, H, D]

        # ----- Edge update (node-conditioned, residual-gated) -----
        # Build [x_src, x_dst, e] in the SAME attention space (in_dim = H*D)
        x_src = x[src]                           # [E, in_dim]
        x_dst = x[dst]                           # [E, in_dim]
        e_in  = edge_attr                        # [E, in_dim]
        edge_cat = torch.cat([x_src, x_dst, e_in], dim=-1)   # [E, 3*in_dim]

        e_delta = self.edge_mlp(edge_cat)        # [E, in_dim]
        gate    = torch.sigmoid(self.edge_gate(e_in))  # [E, in_dim]
        if self.training and self.edge_dropout > 0:
            e_delta = F.dropout(e_delta, p=self.edge_dropout)

        e_new   = e_in + gate * e_delta          # residual-gated
        e_new   = self.edge_ln(e_new)            # stabilize
        e_new   = F.normalize(e_new, dim=-1)     # keep unit-ish norm for cosine losses

        # For compatibility with downstream code that expects [E,H,D],
        # reshape in_dim (=H*D) back to [E,H,D].
        e_out = e_new.view(E, H, D)

        return h_out, e_out

# -------------------------
# Feed-forward & LN helpers
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, dim, hidden, dropout=0.1):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim)
        )
    def forward(self, x): return self.ff(x)

# ---------------------------------------------
# One edge-aware graph transformer block (512d)
# ---------------------------------------------
class EdgeAwareGraphBlock(nn.Module):
    """
    Node features are adapted to attention space via a 768->512 projector.
    Edge features are already 512 (your pipeline). Both streams updated.
    Residuals for edges use gate: e = e + gate * delta_e, then L2-normalize.
    """
    def __init__(self, num_heads=8, attn_dim=512, edge_in_dim=512,
                 ffn_hidden_node=1024, ffn_hidden_edge=1024, dropout=0.1, attn_dropout=0.0):
        super().__init__()
        assert attn_dim % num_heads == 0, "attn_dim must be divisible by num_heads"

        # attention
        self.attn = MultiHeadAttentionLayerWithEdge(
            in_dim=attn_dim, out_dim=attn_dim // num_heads,
            num_heads=num_heads, attn_dropout=attn_dropout
        )

        # norms
        self.norm1_node = nn.LayerNorm(attn_dim)
        self.norm1_edge = nn.LayerNorm(edge_in_dim)

        self.norm2_node = nn.LayerNorm(attn_dim)
        self.norm2_edge = nn.LayerNorm(edge_in_dim)

        # FFNs
        self.ffn_node = FeedForward(attn_dim, ffn_hidden_node, dropout)
        self.ffn_edge = FeedForward(edge_in_dim, ffn_hidden_edge, dropout)

        # edge residual gate
        self.edge_gate1 = nn.Linear(edge_in_dim, edge_in_dim)
        self.edge_gate2 = nn.Linear(edge_in_dim, edge_in_dim)

    def forward(self, node_feat, edge_feat, edge_index):
        """
        node_feat: (N, 512)
        edge_feat: (E, 512)
        edge_index:    (2, E)
        Returns:
            node_feat_768 (updated), edge_feat (updated)
        """

        # ---- attention (edge-aware) ----
        h_attn, e_attn = self.attn(node_feat, edge_index, edge_feat)  # (N,H,D)->(N,512), (E,H,D)->(E,512)
        h_attn = h_attn.reshape(node_feat.size(0), -1)                    # (N,512)
        e_attn = e_attn.reshape(edge_feat.size(0), -1)               # (E,512)

        # ---- node: residual + norm + FFN + residual + norm ----
        node_feat = self.norm1_node(node_feat + h_attn)
        node_out = self.norm2_node(node_feat + self.ffn_node(node_feat))

        # ---- edge: residual(gated) + norm + FFN + residual(gated) + norm ----
        # 1st residual with gate
        gate1 = torch.sigmoid(self.edge_gate1(edge_feat))
        e_mid = edge_feat + gate1 * e_attn
        e_mid = self.norm1_edge(e_mid)

        # 2nd residual with gate (FFN)
        e_delta = self.ffn_edge(e_mid)
        gate2 = torch.sigmoid(self.edge_gate2(e_mid))
        edge_out = e_mid + gate2 * e_delta
        edge_out = self.norm2_edge(edge_out)
        edge_out = F.normalize(edge_out, dim=-1)


        return node_out, edge_out

# ------------------------------------------------------
# Two-layer Graph Transformer + optional global layer
# ------------------------------------------------------

class SceneGraphEdgeNet(nn.Module):
    def __init__(self,
                 L = 2,
                 geo_dim=19,
                 prior_edge_dim=512,
                 node_feat_dim=768,
                 out_dim=512,
                 dropout=0.1,
                 use_trust_gate=True,
                 num_heads=8,
                 ffn_hidden_node=1024,
                 ffn_hidden_edge=1024,
                 attn_dropout=0.0):
        super().__init__()

        self.edge_encoder = EdgeEncoder(
            geo_dim=geo_dim,
            node_dim=node_feat_dim,
            out_dim=out_dim,
            dropout=dropout
        )

        self.node_encoder = NodeEncoder(
            geo_dim=geo_dim,
            clip_dim=node_feat_dim,
            out_dim=out_dim,
            dropout=dropout
        )

        self.graph_transformer_layers = nn.ModuleList(
            [EdgeAwareGraphBlock(
                    num_heads=num_heads,
                    edge_in_dim=out_dim,
                    attn_dim=out_dim,
                    ffn_hidden_node=ffn_hidden_node,
                    ffn_hidden_edge=ffn_hidden_edge,
                    dropout=dropout,
                    attn_dropout=attn_dropout
                    )
             for _ in range(L)])
    # edge_index, edge_init_feat, edge_geom_feat, node_clip_feat, node_geom_feat
    def forward(self, edge_index, edge_init, edge_geo, node_clip_feat, node_geo):
        """
        node_feat:   (N, 768)
        edge_geo:    (E, geo_dim)
        edge_init:   (E, 512) prior edge features (can be zero)
        edge_index:  (2, E)
        known_mask:  (E,) bool tensor, True means we have prior for this edge

        return:
            edge_out: (E, 512) normalized
        """

        node_feat = self.node_encoder(node_geo, node_clip_feat)
        edge_feat, edge_node_feat = self.edge_encoder(edge_geo, node_clip_feat, edge_index)
        for layer in self.graph_transformer_layers:
            node_feat, edge_feat = layer(node_feat, edge_feat, edge_index)
        return edge_feat

