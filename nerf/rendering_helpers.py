import torch


def xyzs_from_z_vals(rays_o, rays_d, z_vals):
    return rays_o.unsqueeze(-2) + rays_d.unsqueeze(-2) * z_vals.unsqueeze(-1)  # [N, 1, 3] * [N, T, 1] -> [N, T, 3]


def unflatten_density_outputs(density_outputs, num_rays, samples_per_ray):
    unflattened_outputs = {}
    for k, v in density_outputs.items():
        unflattened_outputs[k] = v.view(num_rays, samples_per_ray, -1)
    return unflattened_outputs

def flatten_density_outputs(density_outputs):
    density_outputs_flat = {}
    for k, v in density_outputs.items():
        density_outputs_flat[k] = v.view(-1, v.shape[-1])
    return density_outputs_flat


def get_foreground_z_vals(nears, fars, num_steps):
    z_vals = torch.linspace(0.0, 1.0, num_steps, device=nears.device).unsqueeze(0)  # [1, T]
    N = nears.shape[-1]
    z_vals = z_vals.expand((N, num_steps))  # [N, T]
    z_vals = nears.unsqueeze(-1) + (fars - nears).unsqueeze(-1) * z_vals  # [N, T], in [nears, fars]
    return z_vals


def perturb_z_vals(z_vals):
    deltas = z_vals[:, 1:] - z_vals[:, :-1]
    assert (deltas >= 0).all()
    deltas = torch.cat((deltas, deltas[:, -1].unsqueeze(-1)), dim=-1)

    perturbation_amount = deltas * 0.5
    perturbed_z_vals = z_vals + 2. * (torch.rand(z_vals.shape, device=z_vals.device) - 0.5) * perturbation_amount

    # clamp the z vals so that they are not less than the previous
    perturbed_z_vals[:, 1:] = perturbed_z_vals[:, 1:].clamp(min=perturbed_z_vals[:, :-1])

    # clamp the z vals so that they are not greater than the subsequent
    perturbed_z_vals[:, :-1] = perturbed_z_vals[:, :-1].clamp(max=perturbed_z_vals[:, 1:])

    return perturbed_z_vals


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # bins: [B, T], old_z_vals
    # weights: [B, T - 1], bin weights.
    # return: [B, n_samples], new_z_vals

    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples).to(weights.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples]).to(weights.device)

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (B, n_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def upsample_z_values(z_vals, sigmas, upsample_steps: int, deterministic: bool):
    with torch.no_grad():
        deltas = z_vals[..., 1:] - z_vals[..., :-1]  # [N, T-1]
        final_delta = (z_vals[:, -1] - z_vals[:, 0]) / (len(z_vals) - 1)
        deltas = torch.cat([deltas, final_delta.unsqueeze(-1)], dim=-1)

        weights = get_alpha_compositing_weights(deltas=deltas, sigmas=sigmas)

        # sample new z_vals
        z_vals_mid = (z_vals[..., :-1] + 0.5 * deltas[..., :-1])  # [N, T-1]
        new_z_vals = sample_pdf(z_vals_mid, weights[:, 1:-1],
                                upsample_steps, det=deterministic).detach()  # [N, t]
        return new_z_vals


def get_alpha_compositing_weights(deltas, sigmas):
    assert (deltas >= 0.).all()

    # Why the eps below? Because if delta = 0 and sigma = infinity, this guarantees we get zero out of the
    # exponential rather than nan.
    eps = 1e-12
    alphas = 1 - torch.exp(-(deltas + eps) * sigmas.squeeze(-1))  # [N, T+t]
    alphas_shifted = torch.cat([torch.ones_like(alphas[..., :1]), 1 - alphas + 1e-15], dim=-1)  # [N, T+t+1]
    cumulative_alphas = torch.cumprod(alphas_shifted, dim=-1)[..., :-1]  # [N, T+t]
    weights = alphas * cumulative_alphas
    return weights
