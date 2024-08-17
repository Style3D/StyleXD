import argparse
import os

import numpy as np
from tqdm import trange

from datasets import SXDImageDataset
from network import SXDUNet2DModel

import plotly.graph_objects as go

_AVATAR_BBOX = np.array([
    [-449.00006104,  191.10876465, -178.34872437],      # min
    [ 447.45980835, 1831.29016113,  174.13575745]       # max
])

_OFFSET = _AVATAR_BBOX[0] + (_AVATAR_BBOX[1] - _AVATAR_BBOX[0]) / 2
_SCALE = _AVATAR_BBOX[1] - _AVATAR_BBOX[0]

def parse_args():
    
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help=("Checkpoint directory.")
    )
    parser.add_argument(
        "--ckpt_step",
        type=int,
        default=10000,
        help=("Checkpoint step.")
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help=("Path to dataset."),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--in_channels",
        type=int,
        default=16,
        help=(
            "Input channels, i.e. positional encoded 2D coordinates."
        ),
    )
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def main(args):
    expr_name = os.path.basename(args.ckpt_dir)
    out_dir = os.path.join(args.output_dir, expr_name)
    os.makedirs(out_dir, exist_ok=True)

    use_cf, uv_type, reso, _ = expr_name.split('_')
    ckpt_dir = os.path.join(args.ckpt_dir, 'checkpoint-%d'%(args.ckpt_step))
    
    dataset = SXDImageDataset(args.data_dir, reso=int(reso), split='test', uv_type=uv_type)

    model = SXDUNet2DModel(
            sample_size=int(reso),
            in_channels=args.in_channels, 
            num_class_embeds=dataset.num_classes if use_cf == 'cf' else -1, 
            out_channels=3,
            layers_per_block=1,
            block_out_channels=(32, 64, 128, 128),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            )
        )

    load_model = SXDUNet2DModel.from_pretrained(ckpt_dir, subfolder="unet")
    model.register_to_config(**load_model.config)
    model.load_state_dict(load_model.state_dict())
    model.to('cuda')
    
    for idx in trange(len(dataset)):
        geo_arr, seg_arr, uv_arr = dataset[idx]        
        pred = model(uv_arr[None].to('cuda'), seg_arr[None].to('cuda')).sample[0]
        
        np.savez_compressed(
            os.path.join(out_dir, '%04d.npz'%(idx)), 
            geo=geo_arr.detach().cpu().numpy(), 
            seg=seg_arr.detach().cpu().numpy(),
            uv=uv_arr.detach().cpu().numpy(),
            pred=pred.detach().cpu().numpy()
            )
        
        mask_arr = seg_arr > 0
        error_map = np.linalg.norm(geo_arr - pred, axis=0)
        
        gt_pts = geo_arr.reshape(3, -1)[:, mask_arr.reshape(-1)]
        gt_pts = gt_pts * (_SCALE[:, None] * 0.5) + _OFFSET[:, None]
        gt_vis = go.Scatter3d(
            x=gt_pts[0, :],
            y=gt_pts[1, :],
            z=gt_pts[2, :],
            mode='markers',
            marker=dict(
                color='#D3D3D3',  # Light grey for ground truth
                size=1
            ))
        
        pred_pts = pred.reshape(3, -1)[:, mask_arr.reshape(-1)]
        pred_pts = pred_pts * (_SCALE[:, None] * 0.5) + _OFFSET[:, None]
        error_values = error_map.reshape(-1)[mask_arr.reshape(-1)]  # Apply the error map to valid points
        pred_vis = go.Scatter3d(
            x=pred_pts[0, :],
            y=pred_pts[1, :],
            z=pred_pts[2, :],
            mode='markers',
            marker=dict(
                color=error_values,  # Color by error values
                colorscale='Rainbow',  # Choose a colorscale, e.g., Viridis
                size=2,
                colorbar=dict(title="Reconstruction Error"),
                showscale=False,
                cmin=0.0,  # Set the minimum value of the color scale
                cmax=1.0 
            ))

        fig = go.Figure(data=[gt_vis, pred_vis],
            layout=go.Layout(
                width=1024,
                height=1024,
                margin=dict(l=1, r=1, b=1, t=1),
                scene=dict(aspectmode='data',xaxis = dict(visible=False),yaxis = dict(visible=False),zaxis = dict(visible=False)),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
        )

        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=2.0)
        )

        fig.update_layout(scene_camera=camera, title="Reconstructed point cloud.")
        fig.write_image(os.path.join(out_dir, '%04d..png'%(idx)))
            
        print('[DONE] Export testing result to %s.'%(out_dir))
    
    
if __name__ == "__main__":
    args = parse_args()
    main(args)