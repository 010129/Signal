import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenSelection(nn.Module):
    def __init__(self, dim, k=112):
        super().__init__()
        self.dim = dim
        self.k1 = k
        self.k2 = 2*k
        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)


    def intra_modal_token_selection(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
       
        batch_size = rgb_patches.size(0)



        rgb_scores = F.softmax((torch.bmm(rgb_global.unsqueeze(1), rgb_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1),dim=1)  
        nir_scores = F.softmax((torch.bmm(nir_global.unsqueeze(1), nir_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1),dim=1)  
        tir_scores = F.softmax((torch.bmm(tir_global.unsqueeze(1), tir_patches.transpose(1, 2))/math.sqrt(self.dim)).squeeze(1),dim=1)  
        
        rgb_topk, rgb_indices = torch.topk(rgb_scores, min(self.k1, rgb_scores.size(1)), dim=1)
        nir_topk, nir_indices = torch.topk(nir_scores, min(self.k1, nir_scores.size(1)), dim=1)
        tir_topk, tir_indices = torch.topk(tir_scores, min(self.k1, tir_scores.size(1)), dim=1)
        
        
        rgb_mask = torch.zeros_like(rgb_patches[:, :, 0])
        nir_mask = torch.zeros_like(nir_patches[:, :, 0])
        tir_mask = torch.zeros_like(tir_patches[:, :, 0])
        
        
        for i in range(batch_size):
            rgb_mask[i, rgb_indices[i]] = 1
            nir_mask[i, nir_indices[i]] = 1
            tir_mask[i, tir_indices[i]] = 1
        
        return rgb_mask.unsqueeze(-1), nir_mask.unsqueeze(-1), tir_mask.unsqueeze(-1)

    def inter_modal_token_selection(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        batch_size = rgb_patches.size(0)
        n_rgb = rgb_patches.size(1)
        n_nir = nir_patches.size(1)
        
        queries = torch.stack([rgb_global, nir_global, tir_global], dim=1)  
        keys = torch.cat([rgb_patches, nir_patches, tir_patches], dim=1)  
        

        q = self.W_q(queries) 
        k = self.W_k(keys)  
        

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.dim)  
        scores = F.softmax(scores, dim=2)
        

        rgb_scores = torch.cat([
            scores[:, 0, n_rgb:n_rgb+n_nir],  
            scores[:, 0, n_rgb+n_nir:]  
        ], dim=1)
        
        nir_scores = torch.cat([
            scores[:, 1, :n_rgb],  
            scores[:, 1, n_rgb+n_nir:]  
        ], dim=1)
        
        tir_scores = torch.cat([
            scores[:, 2, :n_rgb],  
            scores[:, 2, n_rgb:n_rgb+n_nir]  
        ], dim=1)
        

        rgb_topk, rgb_indices = torch.topk(rgb_scores, min(self.k2, rgb_scores.size(1)), dim=1)
        nir_topk, nir_indices = torch.topk(nir_scores, min(self.k2, nir_scores.size(1)), dim=1)
        tir_topk, tir_indices = torch.topk(tir_scores, min(self.k2, tir_scores.size(1)), dim=1)
        
        
        rgb_mask = torch.zeros_like(rgb_patches[:, :, 0])
        nir_mask = torch.zeros_like(nir_patches[:, :, 0])
        tir_mask = torch.zeros_like(tir_patches[:, :, 0])
        
        
        for i in range(batch_size):
            
            nir_selected = rgb_indices[i] < n_nir
            tir_selected = ~nir_selected
            nir_pos = rgb_indices[i][nir_selected]
            tir_pos = rgb_indices[i][tir_selected] - n_nir
            if nir_pos.numel() > 0:
                nir_mask[i, nir_pos] = 1
            if tir_pos.numel() > 0:
                tir_mask[i, tir_pos] = 1
                
            
            nir_selected = nir_indices[i] < n_rgb
            tir_selected = ~nir_selected
            rgb_pos = nir_indices[i][nir_selected]
            tir_pos = nir_indices[i][tir_selected] - n_rgb
            if rgb_pos.numel() > 0:
                rgb_mask[i, rgb_pos] = 1
            if tir_pos.numel() > 0:
                tir_mask[i, tir_pos] = 1
                
            
            rgb_selected = tir_indices[i] < n_rgb
            nir_selected = ~rgb_selected
            rgb_pos = tir_indices[i][rgb_selected]
            nir_pos = tir_indices[i][nir_selected] - n_rgb
            if rgb_pos.numel() > 0:
                rgb_mask[i, rgb_pos] = 1
            if nir_pos.numel() > 0:
                nir_mask[i, nir_pos] = 1
        
        return rgb_mask.unsqueeze(-1), nir_mask.unsqueeze(-1), tir_mask.unsqueeze(-1)
    
    
    
    def forward(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
       
        rgb_mask_c, nir_mask_c, tir_mask_c = self.inter_modal_token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        
        
        rgb_mask_i, nir_mask_i, tir_mask_i = self.intra_modal_token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        
        
        rgb_mask = (rgb_mask_c + rgb_mask_i > 0).float()
        nir_mask = (nir_mask_c + nir_mask_i > 0).float()
        tir_mask = (tir_mask_c + tir_mask_i > 0).float()
        
        
        rgb_selected = rgb_patches * rgb_mask
        nir_selected = nir_patches * nir_mask
        tir_selected = tir_patches * tir_mask
        
        return rgb_selected, nir_selected, tir_selected

        

class ModalInteractive(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
       
        self.ffn = nn.Sequential(
            nn.Linear(dim, 2*dim),
            nn.GELU(),
            nn.Linear(2*dim, dim)
        )
        
        
        self.norm1 = LayerNorm(dim)
        self.norm2 = LayerNorm(dim)
        
    def forward(self, rgb_selected, nir_selected, tir_selected, rgb_global, nir_global, tir_global):

        queries = torch.stack([rgb_global, nir_global, tir_global], dim=1)  # [bs, 3, dim]
        keys_values = torch.cat([rgb_selected, nir_selected, tir_selected], dim=1)  # [bs, 3*n, dim]


        attn_output, _ = self.cross_attn(queries, keys_values, keys_values)
        
        attn_output = self.norm1(queries + attn_output)
        
        ffn_output = self.ffn(attn_output)
        
        fused_features = self.norm2(attn_output + ffn_output)
        
        final_feature = torch.cat([fused_features[:, 0], fused_features[:, 1], fused_features[:, 2]], dim=1)
        
        return final_feature

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    """before and after attention,use it"""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class Select_Interactive_Module(nn.Module):
    def __init__(self, dim, k=112):
        super().__init__()
        num_heads=8
        self.token_selection = TokenSelection(dim, k)
        self.modal_interactive = ModalInteractive(dim, num_heads)
        
    def forward(self, rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global):
        rgb_selected, nir_selected, tir_selected = self.token_selection(
            rgb_patches, nir_patches, tir_patches, rgb_global, nir_global, tir_global
        )
        
        final_feature = self.modal_interactive(
            rgb_selected, nir_selected, tir_selected, rgb_global, nir_global, tir_global
        )
        

        return final_feature



