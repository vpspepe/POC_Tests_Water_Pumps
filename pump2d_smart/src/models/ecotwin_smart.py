import torch
import torch.nn as nn
from typing import Any, Tuple

from smart.smart.models.smart.smart import SMART

class EcotwinSMART(nn.Module):
    """Wrapper around SMART model with hard physical constraints."""
    
    def __init__(
        self,
        smart_model: SMART,
        use_pressure_constraint: bool = False,
        pressure_target: float = 0.0,
        sdf_outlet_index: int = 2,
        pressure_channel_index: int = 0,
    ):
        super().__init__()
        self.smart = smart_model
        # Automatically infer spatial_dim from underlying SMART model
        self.spatial_dim = getattr(self.smart, "spatial_dim", 2)
        
        self.use_pressure_constraint = use_pressure_constraint
        self.pressure_target = pressure_target
        self.sdf_outlet_index = sdf_outlet_index
        self.pressure_channel_index = pressure_channel_index
        
        # We can use Tanh to saturate the distance. 
        # A learnable alpha controls the saturation speed.
        self.alpha = nn.Parameter(torch.tensor(50.0))
        self.saturation = nn.Tanh()

        # Forward attributes so it looks like the underlying SMART model
        self.surface_channels = self.smart.surface_channels
        self.volume_channels = self.smart.volume_channels
        self.extra_query_dim = self.smart.extra_query_dim
        self.subregion_size = self.smart.subregion_size

    def initialize_weights(self):
        self.smart.initialize_weights()

    def _apply_pressure_constraint(self, n_p: torch.Tensor, d_out: torch.Tensor) -> torch.Tensor:
        """Applies the outlet pressure constraint: p(x) = p_target + tanh(|alpha| * |d_out(x)|) * N_p(x)"""
        if d_out.dim() == 2:
            d_out = d_out.unsqueeze(-1)
        
        dist = torch.abs(d_out)
        alpha = torch.abs(self.alpha)
        
        # Saturated distance term -> 0 at boundary, 1 far away
        return self.pressure_target + self.saturation(alpha * dist) * n_p

    def forward(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        extra_query_features=None,
        d_out=None,
    ):
        pred_surf, pred_vol = self.smart(
            geo, surf_query_pos, vol_query_pos, params, extra_query_features
        )
        
        if self.use_pressure_constraint:
            if d_out is None and extra_query_features is not None and self.sdf_outlet_index >= 0:
                N_surf = surf_query_pos.shape[1]
                if N_surf > 0 and self.surface_channels > 0 and extra_query_features.shape[1] > N_surf:
                    vol_extra = extra_query_features[:, N_surf:, :]
                else:
                    vol_extra = extra_query_features
                if vol_extra.shape[-1] > self.sdf_outlet_index:
                    d_out = vol_extra[..., self.sdf_outlet_index:self.sdf_outlet_index+1]
                    
            if d_out is not None:
                idx = self.pressure_channel_index
                n_p = pred_vol[..., idx:idx+1]
                p = self._apply_pressure_constraint(n_p, d_out)
                if idx == 0:
                    pred_vol = torch.cat([p, pred_vol[..., 1:]], dim=-1)
                else:
                    pred_vol = torch.cat([pred_vol[..., :idx], p, pred_vol[..., idx+1:]], dim=-1)
            
        return pred_surf, pred_vol

    @torch.inference_mode()
    def inference(
        self,
        geo,
        surf_query_pos,
        vol_query_pos,
        params,
        extra_surf_features=None,
        extra_vol_features=None,
        d_out=None,
    ):
        pred_surf, pred_vol = self.smart.inference(
            geo, surf_query_pos, vol_query_pos, params, extra_surf_features, extra_vol_features
        )
        
        if self.use_pressure_constraint:
            if d_out is None and extra_vol_features is not None and self.sdf_outlet_index >= 0:
                if extra_vol_features.shape[-1] > self.sdf_outlet_index:
                    d_out = extra_vol_features[..., self.sdf_outlet_index:self.sdf_outlet_index+1]
            if d_out is not None:
                idx = self.pressure_channel_index
                n_p = pred_vol[..., idx:idx+1]
                p = self._apply_pressure_constraint(n_p, d_out)
                if idx == 0:
                    pred_vol = torch.cat([p, pred_vol[..., 1:]], dim=-1)
                else:
                    pred_vol = torch.cat([pred_vol[..., :idx], p, pred_vol[..., idx+1:]], dim=-1)
            
        return pred_surf, pred_vol
