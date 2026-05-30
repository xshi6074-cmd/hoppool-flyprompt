import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

logger = logging.getLogger()


class Prompt(nn.Module):
    def __init__(self,
                 num_experts: int,
                 len_prompt: int = 20,
                 embed_dim: int = 768,
                 pos_prompt: Iterable[int] = (0, 1, 2, 3, 4)):
        super().__init__()
        self.num_experts = num_experts
        self.len_prompt = len_prompt
        self.embed_dim = embed_dim

        self.register_buffer('pos_prompt', torch.tensor(list(pos_prompt), dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())

        self.prompts = nn.Parameter(
            torch.empty(self.num_layers, num_experts, len_prompt, embed_dim)
        )
        nn.init.uniform_(self.prompts)

    def _build_batched_prompts(self, backbone: nn.Module, expert_ids: torch.Tensor) -> torch.Tensor:
        B = expert_ids.size(0)
        prompts = []
        for l_idx in range(self.num_layers):
            p_l = self.prompts[l_idx][expert_ids.long()]  # [B, len_prompt, D]
            prompts.append(p_l)
        prompts = torch.stack(prompts, dim=1)  # [B, num_layers, len_prompt, D]

        D = prompts.size(-1)
        pos_bias = backbone.pos_embed[:, :1, :].unsqueeze(1).expand(B, self.num_layers, self.len_prompt, D)
        prompts = prompts + pos_bias
        return prompts

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        token_appended = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(token_appended + backbone.pos_embed)
        orig_N = x.size(1)

        prompts = self._build_batched_prompts(backbone, expert_ids)  # [B, num_layers, len_prompt, D]

        for n, block in enumerate(backbone.blocks):
            pos_n = (self.pos_prompt.eq(n)).nonzero(as_tuple=False).squeeze()
            if pos_n.numel() != 0:
                x = torch.cat((x, prompts[:, pos_n]), dim=1)
            x = block(x)
            x = x[:, :orig_N, :]

        x = backbone.norm(x)
        return x[:, 0]

    @torch.no_grad()
    def init_new_expert(self, expert_id: int):
        if expert_id == 0 or expert_id >= self.num_experts:
            return
        prev_experts = self.prompts[:, :expert_id].clone()  # [num_layers, expert_id, L, D]
        prev_experts_mean = prev_experts.mean(dim=1)        # [num_layers, L, D]
        self.prompts.data[:, expert_id] = prev_experts_mean


class RPFC(nn.Module):
    def __init__(self,
                 M            : int,
                 ridge        : float = 1e4,
                 embed_dim    : int = 768,
                 num_classes  : int = 100,
                 **kwargs):

        super().__init__()
        
        self.ridge = ridge
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        if M == 0:
            self.M = embed_dim
            self.use_rp = False
            self.register_buffer('W_rand', torch.empty(0))
            self.register_buffer('Q', torch.zeros(embed_dim, num_classes))
            self.register_buffer('G', torch.zeros(embed_dim, embed_dim))
        else:
            self.M = M
            self.use_rp = True
            self.register_buffer('W_rand', torch.randn(embed_dim, M))
            self.register_buffer('Q', torch.zeros(M, num_classes))
            self.register_buffer('G', torch.zeros(M, M))

        self.fc = nn.Linear(self.M, num_classes, bias=False)

        for param in self.parameters():
            param.requires_grad = False

    def target2onehot(self, targets):
        device = targets.device
        onehot = torch.zeros(targets.size(0), self.num_classes, device=device)
        onehot.scatter_(1, targets.unsqueeze(1), 1)
        return onehot

    def collect(self, features, labels):
        features = features.detach()
        labels = labels.detach()

        if self.use_rp:
            features_h = F.relu(features @ self.W_rand)
        else:
            features_h = features
        Y = self.target2onehot(labels)
        self.Q = self.Q + features_h.T @ Y
        self.G = self.G + features_h.T @ features_h

    def update(self):
        device = self.fc.weight.device
        Wo = torch.linalg.solve(self.G + self.ridge * torch.eye(self.M, device=device), self.Q).T
        self.fc.weight.data = Wo.to(device)

    def forward(self, x):
        if self.use_rp:
            x = F.relu(x @ self.W_rand)
        x = self.fc(x)
        return x


class FlyPrompt(nn.Module):
    def __init__(self,
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 backbone_name  : str   = None,
                 len_prompt     : int   = 20,
                 pos_prompt     : Iterable[int] = (0, 1, 2, 3, 4),
                 rp_dim         : int   = 10000,
                 rp_ridge       : float = 1e4,
                 ema_ratio      : Iterable[float] = (0.9, 0.99),
                 **kwargs):

        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt
        self.rp_dim = rp_dim
        self.rp_ridge = rp_ridge
        self.ema_ratio = ema_ratio
        self.num_ema = len(ema_ratio)

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad   = True

        # Expert prompts
        self.experts = Prompt(
            num_experts = self.task_num,
            len_prompt = self.len_prompt,
            embed_dim = self.embed_dim,
            pos_prompt = self.pos_prompt,
        )

        # Expert FCs
        self.experts_fc = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.num_ema)
            ]) for _ in range(self.task_num)
        ])
        for expert_fc in self.experts_fc:
            for fc in expert_fc:
                for param in fc.parameters():
                    param.requires_grad = False
        self.init_fc(expert_id = 0)

        # Random projection head
        self.rp_head = RPFC(
            M = self.rp_dim,
            ridge = self.rp_ridge,
            embed_dim = self.embed_dim,
            num_classes = self.task_num,
        )

        load_pt = kwargs.get('load_pt', False)
        logger.info(f"FlyPrompt init: load_pt={load_pt}")
        if load_pt:
            self.load_prompt()

    def load_prompt(self):
        g_path = "./checkpoints/g_prompt.pt"
        e_path = "./checkpoints/e_prompt.pt"
        import os
        abs_g_path = os.path.abspath(g_path)
        abs_e_path = os.path.abspath(e_path)
        
        if os.path.exists(abs_g_path) and os.path.exists(abs_e_path):
            logger.info(f"FlyPrompt: Successfully found prompts at {abs_g_path} and {abs_e_path}")
            try:
                # FlyPrompt uses 'experts.prompts' with shape [num_layers, num_experts, len_prompt, embed_dim]
                # g_prompt is likely [10, 768] (pool size, dim) or [layers, pool, len, dim]
                g_weight = torch.load(abs_g_path, map_location='cpu')
                e_weight = torch.load(abs_e_path, map_location='cpu')
                logger.info(f"FlyPrompt: Loaded g_prompt shape: {g_weight.shape}, e_prompt shape: {e_weight.shape}")
                
                # FlyPrompt specifically uses 'experts.prompts'
                # If we want to load into 'experts.prompts', we need to match [num_layers, num_experts, len_prompt, embed_dim]
                target_shape = self.experts.prompts.shape
                # Custom loading logic based on FlyPrompt structure
                # This is a placeholder as FlyPrompt doesn't have a split g/e prompt like DualPrompt
                # But if users want to load MISA prompts into FlyPrompt:
                logger.warning("FlyPrompt: Attempting to map loaded prompts to 'experts.prompts'...")
                
                # For now, let's just log and try a basic assignment if shapes happen to match or can be reshaped
                # Note: MISA prompt structure is different, so this might need adjustment based on actual .pt content
                
            except Exception as e:
                logger.error(f"FlyPrompt: Error during loading: {str(e)}")
        else:
            logger.error(f"FlyPrompt: Prompt files NOT found at {abs_g_path} or {abs_e_path}")

    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x
    
    def forward_with_rp(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.forward_features(inputs)
        x = x[:, 0]
        x = self.rp_head(x)
        return x
    
    def forward_with_ema(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if expert_ids is None:
            expert_ids = torch.full((inputs.size(0),), self.task_count, device=inputs.device, dtype=torch.long)
        x = self.experts(self.backbone, inputs, expert_ids)
        outputs_ls = []

        # online head
        outputs_ls.append(self.backbone.fc(x))
        
        # ema head
        for i in range(self.num_ema):
            outputs = []
            for x_i, e_i in zip(x, expert_ids):
                outputs.append(self.experts_fc[e_i.item()][i](x_i))
            outputs = torch.stack(outputs, dim=0)
            outputs_ls.append(outputs)

        return outputs_ls
    
    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        features = self.backbone.forward_features(inputs)
        features = features[:, 0]
        labels = torch.full((labels.size(0),), self.task_count, device=labels.device, dtype=torch.long)
        self.rp_head.collect(features, labels)

    def update(self):
        self.rp_head.update()

    @torch.no_grad()
    def init_fc(self, expert_id: int = None):
        if expert_id is None:
            expert_id = self.task_count
        if expert_id >= self.task_num:
            return
        w, b = self.backbone.fc.weight.data, self.backbone.fc.bias.data
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int = None):
        if expert_id is None:
            expert_id = self.task_count
        for i in range(self.num_ema):
            ema_ratio = self.ema_ratio[i]
            online_w = self.backbone.fc.weight.data
            online_b = self.backbone.fc.bias.data
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data
            ema_w.mul_(ema_ratio).add_(online_w, alpha=1.0 - ema_ratio)
            ema_b.mul_(ema_ratio).add_(online_b, alpha=1.0 - ema_ratio)

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    def process_task_count(self):
        self.task_count += 1
        self.rp_head.update()
        self.experts.init_new_expert(self.task_count)
        self.init_fc(self.task_count)