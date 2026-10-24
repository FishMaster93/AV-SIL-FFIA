"""
HAIL-FFIR 的音视频融合骨干（论文 Sec. III-B1, Eq. 1-2），作用在冻结编码器的缓存特征上：

  视频 f^v: S3D 时空特征图 (B, L=2, S=49, d=1024)
  音频 f^a: Cnn14-MobileNetV2 fc1 embedding (B, d=1024)

  音频引导的视觉注意力（沿用 AV-CIL [16] 的空间->时间两级注意力）:
      w^Spa_l = softmax_s( <K_s(f^v_{l,s}), Q_s(f^a)> / sqrt(h) )
      v_l     = sum_s w^Spa_{l,s} f^v_{l,s}
      w^Tem   = softmax_l( <K_t(v_l), Q_t(f^a)> / sqrt(h) )
      f'^v    = sum_l w^Tem_l v_l
  视觉引导的音频注意力（论文的双向扩展；通道注意力，softmax 归一化后乘 d 使均值为 1）:
      w^Audio = d * softmax( W_g mean_{l,s}(f^v) )
      f'^a    = f^a ⊙ w^Audio
  融合 (Eq. 2):  f^av = σ(f'^v U^v) + σ(f'^a U^a)，σ 取 sigmoid（与 Eq. 13 同一符号）
  跨模态一致性: L_cos = 1 - cos(f'^v U^v, f'^a U^a)

论文没写死的细节（注意力维度 h、输入 LayerNorm、σ 的具体形式）在这里固定下来，
只在 Phase 0 上训练一次，之后冻结。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BidirectionalCrossModalFusion(nn.Module):
    def __init__(self, d=1024, h=256):
        super().__init__()
        self.ln_v = nn.LayerNorm(d)
        self.ln_a = nn.LayerNorm(d)
        self.q_spa, self.k_spa = nn.Linear(d, h), nn.Linear(d, h)
        self.q_tem, self.k_tem = nn.Linear(d, h), nn.Linear(d, h)
        self.w_g = nn.Linear(d, d)
        nn.init.zeros_(self.w_g.weight)   # 初始 w^Audio = 1（恒等），训练中再学
        nn.init.zeros_(self.w_g.bias)
        self.U_v = nn.Linear(d, d, bias=False)
        self.U_a = nn.Linear(d, d, bias=False)
        self.h = h

    def forward(self, fv, fa):
        """fv: (B, L, S, d)  fa: (B, d) -> dict(f_av, f_v, f_a, z_v, z_a, attn...)"""
        fv = self.ln_v(fv.float())
        fa = self.ln_a(fa.float())
        scale = 1.0 / math.sqrt(self.h)

        # 音频引导的空间注意力，每个时间步一组
        q = self.q_spa(fa)                                             # (B, h)
        k = self.k_spa(fv)                                             # (B, L, S, h)
        w_spa = torch.softmax(torch.einsum('blsh,bh->bls', k, q) * scale, dim=-1)
        v = torch.einsum('bls,blsd->bld', w_spa, fv)                   # (B, L, d)
        # 音频引导的时间注意力
        w_tem = torch.softmax(torch.einsum('blh,bh->bl', self.k_tem(v), self.q_tem(fa)) * scale, dim=-1)
        f_v = torch.einsum('bl,bld->bd', w_tem, v)                     # f'^v

        # 视觉引导的音频通道注意力
        g = fv.mean(dim=(1, 2))
        w_audio = fa.shape[-1] * torch.softmax(self.w_g(g), dim=-1)
        f_a = fa * w_audio                                             # f'^a

        z_v, z_a = self.U_v(f_v), self.U_a(f_a)
        f_av = torch.sigmoid(z_v) + torch.sigmoid(z_a)                 # Eq. 2
        return {'f_av': f_av, 'f_v': f_v, 'f_a': f_a, 'z_v': z_v, 'z_a': z_a,
                'w_spa': w_spa, 'w_tem': w_tem, 'w_audio': w_audio}


def cosine_consistency(z_v, z_a):
    return (1.0 - F.cosine_similarity(z_v, z_a, dim=-1)).mean()
