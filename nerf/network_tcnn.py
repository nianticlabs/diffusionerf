import numpy as np
import tinycudann as tcnn
import torch

from activation import trunc_exp
from .renderer import NeRFRenderer


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=128,
                 bound=1,
                 directional_color_dependence=True,
                 feed_position_to_sigma_net=False,
                 **kwargs
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.directional_color_dependence = directional_color_dependence
        self.feed_position_to_sigma_net = feed_position_to_sigma_net

        base_resolution = 16

        self.finest_resolution = 2048  # unitless! multiply by bound if you want metres or whatever
        self.n_levels = 17
        per_level_scale = np.exp2(
            np.log2(self.finest_resolution * bound / base_resolution) / (self.n_levels - 1)
        )
        self.n_features_per_level = 2
        log2_hashmap_size = 21

        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.n_levels,
                "n_features_per_level": self.n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
        )
        self.grid_spacing_finest = bound / self.finest_resolution

        sigma_net_input_dim = self.n_levels * self.n_features_per_level
        if feed_position_to_sigma_net:
            sigma_net_input_dim += 3

        self.sigma_net = tcnn.Network(
            n_input_dims=sigma_net_input_dim,
            n_output_dims=1 + self.geo_feat_dim,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers - 1,
            },
        )

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        self.encoder_dir = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        if self.directional_color_dependence:
            self.in_dim_color = self.encoder_dir.n_output_dims + self.geo_feat_dim
        else:
            self.in_dim_color = self.geo_feat_dim

        self.color_net = tcnn.Network(
            n_input_dims=self.in_dim_color,
            n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim_color,
                "n_hidden_layers": num_layers_color - 1,
            },
        )

    def forward(self, x, d):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]


        # sigma
        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x = self.encoder(x)
        h = self.sigma_net(x)

        #sigma = F.relu(h[..., 0])
        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
        d = self.encoder_dir(d)

        #p = torch.zeros_like(geo_feat[..., :1]) # manual input padding
        h = torch.cat([d, geo_feat], dim=-1)
        h = self.color_net(h)

        # sigmoid activation for rgb
        color = torch.sigmoid(h)

        return sigma, color

    def density(self, x):
        # x: [N, 3], in [-bound, bound]

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]
        x_enc = self.encoder(x)

        if self.feed_position_to_sigma_net:
            x_enc = torch.cat([x_enc, x], dim=-1)

        h = self.sigma_net(x_enc)

        sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
            'preact_sigma': h[..., 0],
        }

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        x = (x + self.bound) / (2 * self.bound) # to [0, 1]

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        # color
        if self.directional_color_dependence:
            d = (d + 1) / 2 # tcnn SH encoding requires inputs to be in [0, 1]
            d = self.encoder_dir(d)

            h = torch.cat([d, geo_feat], dim=-1)
        else:
            h = geo_feat

        h = self.color_net(h)

        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr},
        ]

        return params

    @property
    def sigma_params(self):
        return self.sigma_net.params
