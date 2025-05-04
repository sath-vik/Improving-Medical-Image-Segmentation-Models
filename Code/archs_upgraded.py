import torch
from torch import nn
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
# Assuming utils.py contains necessary functions if any
# from utils import *

import timm
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.vision_transformer import Block as TimmTransformerBlock
import types
import math
from abc import ABCMeta, abstractmethod
# from mmcv.cnn import ConvModule # Not used in provided snippet
from pdb import set_trace as st

# Assuming kan.py defines KANLinear and KAN
from kan import KANLinear, KAN
from torch.nn import init

__all__ = ['UKAN']

# =============================================================================
# KANLayer, KANBlock, DWConv, DW_bn_relu, PatchEmbed, ConvLayer, D_ConvLayer
# Definitions (Assume these are correct as provided in the original code)
# =============================================================================

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                        in_features, hidden_features, grid_size=grid_size, spline_order=spline_order,
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc2 = KANLinear(
                        hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc3 = KANLinear(
                        hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        # Reshape needed inside KANLinear if it doesn't handle (B,N,C) directly
        # Assuming KANLinear expects (B*N, C) or similar based on original code
        x_reshaped = x.reshape(B * N, C)
        x_reshaped = self.fc1(x_reshaped)
        x = x_reshaped.reshape(B, N, -1) # Reshape back to B, N, HiddenC
        C_hidden = x.shape[-1] # Get hidden dim
        x = self.dwconv_1(x.reshape(B, H, W, C_hidden).permute(0, 3, 1, 2).contiguous(), H, W) # Need correct reshape/permute for DWConv

        x_reshaped = x.reshape(B * N, C_hidden)
        x_reshaped = self.fc2(x_reshaped)
        x = x_reshaped.reshape(B, N, -1) # Reshape back to B, N, OutC
        C_out = x.shape[-1] # Get out dim
        x = self.dwconv_2(x.reshape(B, H, W, C_out).permute(0, 3, 1, 2).contiguous(), H, W) # Need correct reshape/permute

        x_reshaped = x.reshape(B * N, C_out)
        x_reshaped = self.fc3(x_reshaped)
        x = x_reshaped.reshape(B, N, -1)
        x = self.dwconv_3(x.reshape(B, H, W, C_out).permute(0, 3, 1, 2).contiguous(), H, W) # Need correct reshape/permute

        return x # Should return (B, N, C_out)


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, no_kan=False):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim) # Or potentially dim * mlp_ratio like in ViT? Kept as dim based on original.
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop, no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # Same as KANLayer's _init_weights
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # Note: KANLayer might need adjustment if it doesn't handle (B, N, C) input/output correctly
        # The original KANLayer forward seemed to expect reshaping inside.
        # Let's assume KANLayer now correctly processes (B, N, C) -> (B, N, C) possibly using internal reshapes
        x = x + self.drop_path(self.layer(self.norm2(x), H, W))
        return x


class DWConv(nn.Module): # Included for completeness, seems unused by KANLayer/Block
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W): # Input x expected as (B, C, H, W)
        # B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W) # This assumes input is B, N, C
        x = self.dwconv(x)
        # x = x.flatten(2).transpose(1, 2) # Reshape back if needed
        return x

class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W): # Expects (B, C, H, W) input
        # B, N, C = x.shape
        # x = x.transpose(1, 2).view(B, C, H, W) # This assumes input is B, N, C
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # x = x.flatten(2).transpose(1, 2) # Reshape back if needed
        return x

# Inside archs.py

class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.hidden_dim = hidden_features
        self.out_dim = out_features
        self.no_kan = no_kan # Store this

        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        # Use KANLinear or nn.Linear
        if not no_kan:
            self.fc1 = KANLinear(
                        in_features, hidden_features, grid_size=grid_size, spline_order=spline_order,
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc2 = KANLinear(
                        hidden_features, out_features, grid_size=grid_size, spline_order=spline_order,
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
            self.fc3 = KANLinear(
                        out_features, out_features, grid_size=grid_size, spline_order=spline_order, # Note: Changed hidden->out to out->out for fc3 to match dwconv3 dim
                        scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                        base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range)
        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(out_features, out_features) # Note: Changed hidden->out to out->out for fc3

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(out_features) # Match fc2 output dim
        self.dwconv_3 = DW_bn_relu(out_features) # Match fc3 output dim

        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, KANLinear):
             pass # KANLinear might have its own init
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # --- CORRECTED FORWARD METHOD ---
    def forward(self, x, H, W):
        B, N, C = x.shape
        # Input x is (B, N, C)

        # Layer 1
        # Reshape for fc1
        x_reshaped = x.reshape(B * N, C)
        if self.no_kan: # Standard Linear
            x_processed = self.fc1(x_reshaped)
        else: # KANLinear expects 2D input based on the error
            x_processed = self.fc1(x_reshaped)
        # Reshape back to (B, N, hidden_dim)
        x = x_processed.reshape(B, N, self.hidden_dim)

        # Reshape and permute for dwconv_1: (B, N, C) -> (B, C, N) -> (B, C, H, W)
        x_conv = x.transpose(1, 2).reshape(B, self.hidden_dim, H, W)
        x_conv = self.dwconv_1(x_conv, H, W) # dwconv expects (B, C, H, W)
        # Reshape back to tokens: (B, C, H*W) -> (B, N, C)
        x = x_conv.reshape(B, self.hidden_dim, N).transpose(1, 2)

        # Layer 2
        # Reshape for fc2
        x_reshaped = x.reshape(B * N, self.hidden_dim)
        if self.no_kan:
            x_processed = self.fc2(x_reshaped)
        else:
            x_processed = self.fc2(x_reshaped)
        # Reshape back to (B, N, out_dim)
        x = x_processed.reshape(B, N, self.out_dim)

        # Reshape and permute for dwconv_2
        x_conv = x.transpose(1, 2).reshape(B, self.out_dim, H, W)
        x_conv = self.dwconv_2(x_conv, H, W)
        # Reshape back to tokens
        x = x_conv.reshape(B, self.out_dim, N).transpose(1, 2)

        # Layer 3
        # Reshape for fc3
        x_reshaped = x.reshape(B * N, self.out_dim)
        if self.no_kan:
            x_processed = self.fc3(x_reshaped)
        else:
            x_processed = self.fc3(x_reshaped)
        # Reshape back to (B, N, out_dim)
        x = x_processed.reshape(B, N, self.out_dim)

        # Reshape and permute for dwconv_3
        x_conv = x.transpose(1, 2).reshape(B, self.out_dim, H, W)
        x_conv = self.dwconv_3(x_conv, H, W)
        # Reshape back to tokens
        x = x_conv.reshape(B, self.out_dim, N).transpose(1, 2)

        x = self.drop(x)
        # The residual connection is handled in KANBlock
        return x # Output shape (B, N, out_dim)

# Inside archs.py

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding """
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # Ensure stride is also a tuple for consistent calculations
        stride = to_2tuple(stride)

        self.img_size = img_size
        self.patch_size = patch_size

        # --- CORRECTED Grid Size Calculation ---
        # Calculate padding assuming 'same' padding behavior is desired for stride > 1
        # Or simply use the kernel_size//2 padding convention
        padding = (patch_size[0] // 2, patch_size[1] // 2)
        padding = to_2tuple(padding)

        # Formula for Conv2d output size: O = floor((I - K + 2P) / S) + 1
        self.H = (img_size[0] - patch_size[0] + 2 * padding[0]) // stride[0] + 1
        self.W = (img_size[1] - patch_size[1] + 2 * padding[1]) // stride[1] + 1
        self.num_patches = self.H * self.W
        # ---------------------------------------

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=padding)
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights) # Apply initialization

    # Keep the _init_weights and forward methods as they were

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape # Get actual H, W after convolution
        x = x.flatten(2).transpose(1, 2) # B, N, C where N = H * W
        x = self.norm(x)
        return x, H, W # Return tokens and actual H, W


class ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(ConvLayer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, input):
        return self.conv(input)

class D_ConvLayer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(D_ConvLayer, self).__init__()
        # Original used in_ch -> in_ch, then in_ch -> out_ch. Combining for simplicity unless specific reason exists.
        self.conv = nn.Sequential(
             nn.Conv2d(in_ch, out_ch, 3, padding=1), # Modified: Direct in->out
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True),
             nn.Conv2d(out_ch, out_ch, 3, padding=1),
             nn.BatchNorm2d(out_ch),
             nn.ReLU(inplace=True)
         )
        # Original Implementation (if needed):
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_ch, in_ch, 3, padding=1),
        #     nn.BatchNorm2d(in_ch),
        #     nn.ReLU(inplace=True))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, 3, padding=1),
        #     nn.BatchNorm2d(out_ch),
        #     nn.ReLU(inplace=True))
    def forward(self, input):
        # return self.conv2(self.conv1(input)) # Original
        return self.conv(input) # Simplified


# =============================================================================
# UKAN Class - CORRECTED
# =============================================================================

class UKAN(nn.Module):
    def __init__(self, num_classes, input_channels=3, deep_supervision=False, # deep_supervision not used currently
                 img_size=224, patch_size=16, embed_dims=[128, 256, 512], # Example dims
                 no_kan=False, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 2, 1], # Example depths for KAN/Transformer blocks
                 # --- Transformer Bottleneck Parameters ---
                 transformer_depth=2, transformer_num_heads=8, transformer_mlp_ratio=4.,
                 transformer_qkv_bias=True, transformer_drop=0., transformer_attn_drop=0.,
                 # --- End ---
                 **kwargs):
        super().__init__()

        # ---- Input Checks ----
        if len(depths) != 3:
            raise ValueError("depths must have 3 elements (for KAN stage 1, Transformer bottleneck, KAN stage 2)")
        if len(embed_dims) != 3:
            raise ValueError("embed_dims must have 3 elements (output of conv stage 3, KAN stage 1, Bottleneck)")

        # ---- Convolutional Encoder ----
        # Use embed_dims[0] as the target dimension after convolutions
        self.encoder1 = ConvLayer(input_channels, embed_dims[0] // 4) # More gradual increase
        self.encoder2 = ConvLayer(embed_dims[0] // 4, embed_dims[0] // 2)
        self.encoder3 = ConvLayer(embed_dims[0] // 2, embed_dims[0])

        # ---- Patch Embeddings ----
        # Patch embed before first KAN/Transformer stage
        self.patch_embed3 = PatchEmbed(img_size=img_size // 8, # After 3 maxpools (2*2*2=8)
                                       patch_size=3, stride=2, in_chans=embed_dims[0],
                                       embed_dim=embed_dims[1])
        # Patch embed before bottleneck stage
        self.patch_embed4 = PatchEmbed(img_size=img_size // 16, # After patch_embed3 stride 2
                                       patch_size=3, stride=2, in_chans=embed_dims[1],
                                       embed_dim=embed_dims[2])

        # ---- Positional Embedding for Transformer ----
        # Must be defined *after* patch_embed4 to know num_patches
        self.num_patches_bottleneck = self.patch_embed4.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches_bottleneck, embed_dims[2]))
        trunc_normal_(self.pos_embed, std=.02) # Initialize pos_embed

        # ---- Stochastic Depth ----
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        encoder_kan_dpr = dpr[0 : depths[0]]
        bottleneck_dpr  = dpr[depths[0] : depths[0] + depths[1]]
        decoder_kan1_dpr = dpr[depths[0] + depths[1] : depths[0] + depths[1] + depths[2]//2] # Split remaining dpr
        decoder_kan2_dpr = dpr[depths[0] + depths[1] + depths[2]//2 : sum(depths)]

        # ---- KAN/Transformer Blocks ----
        # KAN Block after patch_embed3
        self.block1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=encoder_kan_dpr[i],
            norm_layer=norm_layer, no_kan=no_kan
            ) for i in range(depths[0])])
        self.norm3 = norm_layer(embed_dims[1]) # Norm after KAN stage

        # Transformer Bottleneck Block after patch_embed4
        self.transformer_bottleneck = nn.ModuleList([
            TimmTransformerBlock(
                dim=embed_dims[2], num_heads=transformer_num_heads, mlp_ratio=transformer_mlp_ratio,
                qkv_bias=transformer_qkv_bias, proj_drop=transformer_drop, attn_drop=transformer_attn_drop,
                drop_path=bottleneck_dpr[i], norm_layer=norm_layer, act_layer=nn.GELU
            ) for i in range(depths[1])]) # Use transformer_depth here
        self.norm4 = norm_layer(embed_dims[2]) # Norm after Transformer stage

        # ---- Convolutional Decoder ----
        self.decoder1 = D_ConvLayer(embed_dims[2], embed_dims[1]) # From bottleneck dim to KAN dim

        # ---- Decoder KAN Blocks ----
        # KAN block after decoder1 + skip connection
        self.dblock1 = nn.ModuleList([KANBlock(
            dim=embed_dims[1], drop=drop_rate, drop_path=decoder_kan1_dpr[i],
            norm_layer=norm_layer, no_kan=no_kan
            ) for i in range(depths[2] // 2)]) # Example: half depth here
        self.dnorm3 = norm_layer(embed_dims[1]) # Norm after first decoder KAN stage

        self.decoder2 = D_ConvLayer(embed_dims[1], embed_dims[0]) # From KAN dim to Conv dim

        # KAN block after decoder2 + skip connection
        self.dblock2 = nn.ModuleList([KANBlock(
            dim=embed_dims[0], drop=drop_rate, drop_path=decoder_kan2_dpr[i],
            norm_layer=norm_layer, no_kan=no_kan
            ) for i in range(depths[2] - depths[2] // 2)]) # Example: remaining depth here
        self.dnorm4 = norm_layer(embed_dims[0]) # Norm after second decoder KAN stage

        # Final Convolutional Upsampling Layers
        self.decoder3 = D_ConvLayer(embed_dims[0], embed_dims[0] // 2) # Match encoder dims
        self.decoder4 = D_ConvLayer(embed_dims[0] // 2, embed_dims[0] // 4) # Match encoder dims
        self.decoder5 = D_ConvLayer(embed_dims[0] // 4, embed_dims[0] // 8) # Extra fine level

        # ---- Final Output Layer ----
        self.final = nn.Conv2d(embed_dims[0] // 8, num_classes, kernel_size=1)
        # self.soft = nn.Softmax(dim =1) # Softmax is usually applied outside the model (e.g., in loss function like CrossEntropyLoss)

        # ---- Weight Initialization ----
        self.apply(self._init_weights) # Apply custom init to relevant modules

    def _init_weights(self, m):
        # Specific function to initialize weights like Linear, Conv2d
        # KANLinear/KANBlock might have their own init or rely on their internal modules' init
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
        # Note: We already initialize self.pos_embed separately in __init__


    def forward(self, x):
        B = x.shape[0]

        # === Encoder ===
        # -- Conv Stage --
        # Stage 1
        e1 = self.encoder1(x)
        e1_pool = F.max_pool2d(e1, 2, 2)
        # Stage 2
        e2 = self.encoder2(e1_pool)
        e2_pool = F.max_pool2d(e2, 2, 2)
        # Stage 3
        e3 = self.encoder3(e2_pool)
        e3_pool = F.max_pool2d(e3, 2, 2) # Output: B, C0, H/8, W/8 (C0=embed_dims[0])

        # -- Tokenized KAN Stage --
        # Stage 4 - KAN Blocks
        kan1_in, H1, W1 = self.patch_embed3(e3_pool) # Output: B, N1, C1 (C1=embed_dims[1])
        for blk in self.block1:
            kan1_in = blk(kan1_in, H1, W1)
        kan1_out_norm = self.norm3(kan1_in)
        # Reshape for skip connection and next stage: B, C1, H1, W1 (H1=H/16, W1=W/16)
        kan1_out_img = kan1_out_norm.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # === Bottleneck ===
        # -- Transformer Blocks --
        # Input: kan1_out_img (B, C1, H1, W1)
        trans_in, H_t, W_t = self.patch_embed4(kan1_out_img) # Output: B, N_t, C2 (C2=embed_dims[2], N_t=num_patches_bottleneck)

        # Add Positional Embedding
        trans_in = trans_in + self.pos_embed

        # Apply Transformer Blocks
        for blk in self.transformer_bottleneck:
            trans_in = blk(trans_in)
        trans_out_norm = self.norm4(trans_in)
        # Reshape for Decoder: B, C2, H_t, W_t (H_t=H/32, W_t=W/32)
        bottleneck_out = trans_out_norm.reshape(B, H_t, W_t, -1).permute(0, 3, 1, 2).contiguous()

        # === Decoder ===
        # -- Stage 1 -- KAN Block + Conv Upsample
        # Input: bottleneck_out (B, C2, H_t, W_t)
        d1 = F.interpolate(self.decoder1(bottleneck_out), size=(H1, W1), mode='bilinear', align_corners=False) # Upsample to H1, W1; Output: B, C1, H1, W1
        d1 = torch.add(d1, kan1_out_img) # Skip connection 1 (features before bottleneck)
        # Convert back to tokens for dblock1
        d1_tokens = d1.flatten(2).transpose(1, 2) # B, N1, C1
        for blk in self.dblock1:
            d1_tokens = blk(d1_tokens, H1, W1)
        d1_out_norm = self.dnorm3(d1_tokens)
        # Reshape for next stage: B, C1, H1, W1
        d1_out_img = d1_out_norm.reshape(B, H1, W1, -1).permute(0, 3, 1, 2).contiguous()

        # -- Stage 2 -- KAN Block + Conv Upsample
        # Input: d1_out_img (B, C1, H1, W1)
        H0, W0 = e3.shape[2], e3.shape[3] # Target size H/8, W/8
        d2 = F.interpolate(self.decoder2(d1_out_img), size=(H0, W0), mode='bilinear', align_corners=False) # Upsample to H0, W0; Output: B, C0, H0, W0
        d2 = torch.add(d2, e3) # Skip connection 2 (features before first KAN stage)
        # Convert back to tokens for dblock2
        d2_tokens = d2.flatten(2).transpose(1, 2) # B, N0, C0
        for blk in self.dblock2:
            d2_tokens = blk(d2_tokens, H0, W0)
        d2_out_norm = self.dnorm4(d2_tokens)
        # Reshape for next stage: B, C0, H0, W0
        d2_out_img = d2_out_norm.reshape(B, H0, W0, -1).permute(0, 3, 1, 2).contiguous()

        # -- Final Conv Upsampling Stages --
        # Input: d2_out_img (B, C0, H/8, W/8)
        d3 = F.interpolate(self.decoder3(d2_out_img), size=(e2.shape[2], e2.shape[3]), mode='bilinear', align_corners=False) # Upsample to H/4, W/4
        d3 = torch.add(d3, e2) # Skip connection 3
        # Input: d3 (B, C0/2, H/4, W/4)
        d4 = F.interpolate(self.decoder4(d3), size=(e1.shape[2], e1.shape[3]), mode='bilinear', align_corners=False) # Upsample to H/2, W/2
        d4 = torch.add(d4, e1) # Skip connection 4
        # Input: d4 (B, C0/4, H/2, W/2)
        d5 = F.interpolate(self.decoder5(d4), size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False) # Upsample to H, W

        # === Output ===
        out = self.final(d5)

        return out