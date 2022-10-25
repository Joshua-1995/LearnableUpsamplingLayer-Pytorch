import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LearnableUpsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels=192, out_channels=192, attn_dims=4, content_dims=2):
        super(LearnableUpsamplingLayer, self).__init__()
        
        self.proj_in = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_channels, in_channels)),
            ("swish", nn.SiLU())
        ]))

        self.conv_norm_swish = nn.Sequential(OrderedDict([
             ("conv_ln", ConvLN(in_channels, 8, kernel_size=3, padding=1)),
             ("swish_2", nn.SiLU())
        ]))

        self.W_proj = nn.Sequential(OrderedDict([
             ("linaer_1", nn.Linear(10,
                                   10)),
             ("swish", nn.SiLU()),
             ("linear_2", nn.Linear(10, attn_dims)),
             ("softmax", nn.Softmax(dim=2))
        ]))

        self.C_proj = nn.Sequential(OrderedDict([
            ("linaer_1", nn.Linear(10,
                                   10)),
             ("swish_1", nn.SiLU()),
             ("linear_2", nn.Linear(10,content_dims)),
             ("swish_2", nn.SiLU())
        ]))

        self.proj_WH = nn.Linear(attn_dims*in_channels, out_channels)
        self.proj_WC = nn.Linear(attn_dims*content_dims, out_channels)
        
        self.proj_out= nn.Conv1d(out_channels, out_channels, 1)

    def forward(self, H, d, src_mask):
        '''
        H : phoneme hidden sequence ; [b, h_t=192, t_t]
        d : phoneme duration ; [b, t_t]
        src_mask : phoneme-level mask ; [b, 1, t_text]
        mel_mask : frame-level mask ; [b, 1, t_mel]
        '''
        src_mask = src_mask.unsqueeze(1)
        
        S, E, mel_mask = self.token_boundary_grid(d, src_mask) # [b, t_s, t_t, 1]
        b, t_text, t_mel = src_mask.shape[0], src_mask.shape[-1], mel_mask.shape[-1]

        x = torch.transpose(H, 1, 2)
        x = self.proj_in(x) # [b, t_t, h_t]
        x = torch.transpose(x, 1, 2)
        x = self.conv_norm_swish(x) # [b, 8, t_t]
        
        x = x.unsqueeze(1) # [b, 1, 8, t_t]
        x = torch.repeat_interleave(x, mel_mask.shape[-1], dim=1) # [b, t_s, 8, t_t]
        x = torch.transpose(x, 2, 3) # [b, t_s, t_t, 8]
        x = torch.cat((S, E, x), dim = 3) # [b, t_s, t_t, 10]
    
        W = self.W_proj(x) # [b, t_s, t_t, attn_dims = 4]
        W = W.permute(0,3,1,2) # [b, attn_dims, t_s, t_t]
        C = self.C_proj(x) # [b, t_s, t_t, content_dims = 2]

        WC = torch.einsum('bqmn,bmnp->bmqp',W,C)
        WC = WC.view(b, t_mel, -1) # [b, t_s, attn_dims * content_dims]
        WC = self.proj_WC(WC) # [b, t_s, out_channels]

        WH = torch.einsum('bqmn,bhn->bmqh',W,H)
        WH = WH.view(b, t_mel, -1) # [b, t_s, attn_dims * in_channels]
        WH = self.proj_WH(WH) # [b, t_s, out_channels]

        O = WC + WH # [b, t_s, out_channels]

        O = torch.transpose(O, 1, 2)

        O = self.proj_out(O) * mel_mask

        return O, ~mel_mask.squeeze()

    def token_boundary_grid(self, dur, src_mask):


        mel_len = torch.sum(dur, 1).long()
        max_mel_len = torch.max(mel_len).long()

        mel_mask = self.get_mask_from_lengths(mel_len)
        mel_mask = mel_mask.unsqueeze(1)
        b, t_text, t_mel = src_mask.shape[0], src_mask.shape[-1], mel_mask.shape[-1]

        token_boundary_mask = (torch.unsqueeze(src_mask, 2) * torch.unsqueeze(~mel_mask, -1)).squeeze()
        
        i = torch.arange(1, max_mel_len + 1).unsqueeze(0).to(dur.device)
        i = torch.repeat_interleave(i, b, dim=0).unsqueeze(-1)

        S_d = torch.cat((torch.zeros(b,1).to(dur.device),dur[:,:-1]), dim=1)
        S_d = torch.cumsum(S_d, dim=1).unsqueeze(1)
        S_d = torch.repeat_interleave(S_d, max_mel_len, dim=1)
        S_d = S_d.view(b, t_mel, t_text)

        E_d = torch.cumsum(dur, dim=1).unsqueeze(1)
        E_d = torch.repeat_interleave(E_d, max_mel_len, dim=1)
        E_d = E_d.view(b, t_mel, t_text)
        
        S = (i - S_d) * token_boundary_mask
        E = (E_d - i) * token_boundary_mask

        return S.unsqueeze(-1), E.unsqueeze(-1), ~mel_mask

    def get_mask_from_lengths(self, lengths, max_len=None):
        batch_size = lengths.shape[0]
        if max_len is None:
            max_len = torch.max(lengths).item()
    
        ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
        mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

        return mask

class ConvLN(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size, ln_channels=8, eps=1e-5, padding=1.):
    super().__init__()
    self.channels = ln_channels
    self.eps = eps
    self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)

    self.gamma = nn.Parameter(torch.ones(ln_channels))
    self.beta = nn.Parameter(torch.zeros(ln_channels))

  def forward(self, x):
    x = self.conv(x)
    x = x.transpose(1, -1)
    x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
    return x.transpose(1, -1)
