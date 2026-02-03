from typing import Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat

from layers.InceptionEmbedding import InceptionEmbedding


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.mean(x.pow(2), dim=-1, keepdim=True)
        return self.weight * x * torch.rsqrt(norm + self.eps)


class FFTFrequencyWeight(nn.Module):
    def __init__(self, seq_len, weight_type="learnable"):
        super().__init__()
        self.seq_len = seq_len
        self.weight_type = weight_type
        size = seq_len // 2 + 1

        if weight_type == "learnable":
            self.weight = nn.Parameter(torch.ones(size))
        elif weight_type == "lowpass":
            freqs = torch.arange(size) / size
            self.register_buffer("weight", 1.0 / (1.0 + 10 * freqs))
        elif weight_type == "highpass":
            freqs = torch.arange(size) / size
            self.register_buffer("weight", freqs)
        else:
            self.register_buffer("weight", torch.ones(size))

    def forward(self, x):
        B, C, L = x.shape
        x_fft = torch.fft.rfft(x, dim=-1)
        weight = F.softplus(self.weight) if self.weight_type == "learnable" else self.weight
        x_fft = x_fft * weight.view(1, 1, -1)
        return torch.fft.irfft(x_fft, n=L, dim=-1)


class TransformerBackbone(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.enc_in = config.enc_in
        self.seq_len = config.seq_len
        self.embed_dim = config.d_model
        self.num_heads = config.n_heads
        self.num_layers = config.e_layers
        self.dropout = config.dropout

        self.use_fft_weight = getattr(config, "use_fft_weight", True)
        self.fft_weight_type = getattr(config, "fft_weight_type", "learnable")
        if self.use_fft_weight:
            self.fft_weight = FFTFrequencyWeight(self.seq_len, self.fft_weight_type)

        self.input_proj = InceptionEmbedding(self.enc_in, self.embed_dim)

        self.pos_encoding = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        nn.init.normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=self.num_heads,
            dim_feedforward=self.embed_dim * 4,
            dropout=self.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, self.num_layers)

    def forward(self, x):
        x = x.transpose(1, 2)
        if self.use_fft_weight:
            x = self.fft_weight(x)

        x = self.input_proj(x).transpose(1, 2)
        x = x + self.pos_encoding

        cls = repeat(self.cls_token, "1 1 d -> b 1 d", b=x.shape[0])
        x = torch.cat([cls, x], dim=1)

        x = self.transformer(x)
        return x[:, 0]


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_class = config.num_class
        self.embed_dim = config.d_model
        self.temperature = getattr(config, "temperature", 1.0)

        self.backbone = TransformerBackbone(config)

        self.k_levels = getattr(config, "k_levels", [3,2])
        self.weights = [0.1, 0.25]
        self.counts = [self.num_class * k for k in self.k_levels]

        self.prototype_layers = nn.ParameterList()
        for k in self.k_levels:
            proto = nn.Parameter(
                self.init_prototypes(self.num_class, k, self.embed_dim, radius=3.0),
                requires_grad=False
            )
            self.prototype_layers.append(proto)

        self.dB = 1

    def forward(self, x, _, labels, _a):
        features = self.backbone(x)

        stage_logits = []
        stage_responses = []

        for i, proto in enumerate(self.prototype_layers):
            logits, resp = self._get_stage_logits(features, proto, self.k_levels[i])
            stage_logits.append(logits)
            stage_responses.append(resp)

        if self.training and labels is not None:
            targets = labels.long().squeeze(-1)
            self._update_all_hierarchy(features, targets)
            return stage_responses[-1], self.get_loss(stage_responses, targets)

        return stage_responses[-1]

    def get_loss(self, stage_responses, targets):
        total_loss = 0
        diversity_weight = 0.2
        criterion = nn.CrossEntropyLoss()

        for i, proto in enumerate(self.prototype_layers):
            cls_loss = criterion(stage_responses[i], targets)
            div_loss = self._diversity_loss(proto, self.k_levels[i])
            total_loss += self.weights[i] * cls_loss + diversity_weight * div_loss

        return total_loss

    def _get_stage_logits(self, features, prototypes, k):
        feat = F.normalize(features, dim=-1)
        proto = F.normalize(prototypes, dim=-1)
        logits = feat @ proto.t() / self.temperature
        B = logits.shape[0]
        logits = logits.view(B, self.num_class, k)
        resp = torch.logsumexp(logits, dim=-1)
        return logits.view(B, -1), resp

    def _diversity_loss(self, prototypes, k):
        proto = F.normalize(prototypes, dim=-1)
        proto = proto.view(self.num_class, k, -1)
        sim = torch.bmm(proto, proto.transpose(1, 2))
        eye = torch.eye(k, device=proto.device).unsqueeze(0)
        return F.mse_loss(sim, eye.expand_as(sim))

    def init_prototypes(self, num_class, k, dim, radius):
        protos = torch.zeros(num_class * k, dim)
        for c in range(num_class):
            q, _ = torch.linalg.qr(torch.randn(dim, k))
            q = F.normalize(q.T, dim=-1) * radius
            protos[c * k:(c + 1) * k] = q
        return protos

    @torch.no_grad()
    def _update_all_hierarchy(self, features, labels, alpha=0.5):
        gamma = self._get_warmup_gamma(self.dB)
        self.dB += 1

        centers = []

        proto = self.prototype_layers[0]
        k = self.k_levels[0]
        new_proto, center = self._update_single(proto, features, labels, k)
        proto.data.copy_(gamma * proto.data + (1 - gamma) * new_proto)
        centers.append(center)

        for i in range(1, len(self.prototype_layers)):
            proto = self.prototype_layers[i]
            k = self.k_levels[i]
            new_proto, center = self._update_single(proto, features, labels, k)
            constraint = centers[-1].repeat_interleave(k, dim=0)
            mixed = alpha * new_proto + (1 - alpha) * constraint
            proto.data.copy_(gamma * proto.data + (1 - gamma) * mixed)
            centers.append(center)

    def _update_single(self, prototypes, features, labels, k):
        B, D = features.shape
        proto_dir = F.normalize(prototypes, dim=-1)
        feat_dir = F.normalize(features, dim=-1)
        proto_new = torch.zeros_like(prototypes)

        for c in range(self.num_class):
            mask = labels == c
            if mask.sum() == 0:
                proto_new[c * k:(c + 1) * k] = prototypes[c * k:(c + 1) * k]
                continue

            f = feat_dir[mask]
            p = proto_dir[c * k:(c + 1) * k]
            Q = torch.softmax(f @ p.t(), dim=1)
            raw = features[mask]
            proto_new[c * k:(c + 1) * k] = (Q.t() @ raw) / (Q.sum(0).unsqueeze(1) + 1e-8)

        centers = proto_new.view(self.num_class, k, D).mean(dim=1)
        return proto_new, centers

    def _get_warmup_gamma(self, step, warm=3, active=8):
        if step < warm:
            return 1.0
        if step < warm + active:
            p = (step - warm) / active
            return 1.0 - (1.0 - 0.99) * p
        return 0.99 + (0.999 - 0.99) * (1.0 - math.exp(-(step - warm - active) / 30))
