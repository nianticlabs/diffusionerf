#!/bin/bash

mkdir ./runs/
python -m nerf.evaluate ./data/nerf_llff_data/ \
    --mode blender \
    --upsample_steps 128 \
    --num_rays 1024 \
    --max_ray_batch 1024 \
    --mode llff \
    --downsample_val 8 \
    --reg_ramp_start_step 3000 \
    --reg_ramp_finish_step 8000 \
    --iters 10000 \
    --min_near 0.2 \
    --spread_loss_strength 1.5e-5 \
    --num_train 3 \
    --normalise_length_scales_to_at_least 7.5 \
    --use_frustum_reg \
    --patch_weight_start 0.2 \
    --patch_weight_finish 0.2 \
    --normalise_diffusion_losses \
    --patch_regulariser_path models/rgbd-patch-diffusion.pt \
    --frustum_regularise_patches \
    --apply_geom_reg_to_patches \
    --downsample_train 8 \
    --patch_reg_start_step 0 \
    --only_run_on room \
    --workspace ./runs/example
