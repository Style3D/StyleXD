import json
import os
import numpy as np

import open3d as o3d

import pickle

import torch
torch.set_grad_enabled(False)

import argparse

import imageio
import numpy as np
import nvdiffrast.torch as dr

from glob import glob
from tqdm import tqdm

from matplotlib import pyplot as plt
from matplotlib.colors import to_rgba

from obj_helper import read_obj

from concurrent.futures import ThreadPoolExecutor, as_completed


_CMAP = {
    "帽": {"alias": "hat", "color": "#F7815D"},
    "领": {"alias": "collar", "color": "#F9D26D"},
    "肩": {"alias": "shoulder", "color": "#F23434"},
    "袖片": {"alias": "sleeve", "color": "#C4DBBE"},
    "袖口": {"alias": "cuff", "color": "#F0EDA8"},
    "衣身前中": {"alias": "body front", "color": "#8CA740"},
    "衣身后中": {"alias": "body back", "color": "#4087A7"},
    "衣身侧": {"alias": "body side", "color": "#DF7D7E"},
    
    "底摆": {"alias": "hem", "color": "#DACBBD"},
    "腰头": {"alias": "belt", "color": "#DABDD1"},
    "裙前中": {"alias": "skirt front", "color": "#46B974"},
    "裙后中": {"alias": "skirt back", "color": "#6B68F5"},
    "裙侧": {"alias": "skirt side", "color": "#D37F50"},

    "橡筋": {"alias":"ruffles", "color": "#A8D4D2"},
    "木耳边": {"alias":"ruffles", "color": "#A8D4D2"},
    "袖笼拼条": {"alias":"ruffles", "color": "#A8D4D2"},
    "荷叶边": {"alias":"ruffles", "color": "#A8D4D2"},
    "绑带": {"alias":"ruffles", "color": "#A8D4D2"}
}

_PANEL_CLS = [
    '帽','领','肩','袖片','袖口','衣身前中','衣身后中','衣身侧','底摆','腰头','裙前中','裙后中','裙侧','橡筋','木耳边','袖笼拼条','荷叶边','绑带']
# _PANEL_COLORS = torch.tensor(
#     [to_rgba(_CMAP[_PANEL_CLS[idx]]['color']) for idx in range(len(_PANEL_CLS))],
#     device='cuda', dtype=torch.float32, requires_grad=False)

_PANEL_COLORS = np.array(
    [(0., 0., 0., 0.)] + [to_rgba(_CMAP[_PANEL_CLS[idx]]['color']) for idx in range(len(_PANEL_CLS))]
)

_AVATAR_BBOX = np.array([
    [-449.00006104,  191.10876465, -178.34872437],      # min
    [ 447.45980835, 1831.29016113,  174.13575745]       # max
])

_GLOBAL_MEAN = torch.tensor([-7.6545335e-02, 9.6342279e+02, 5.2539678e+00], dtype=torch.float32, device='cuda')
_GLOBAL_STD = torch.tensor([145.54174098, 308.06958693, 91.57166925], dtype=torch.float32, device='cuda')

_GLOBAL_SCALE = 1000.0
_GLOBAL_OFFSET = np.array([0.0, 1000.0, 0.0])


def to_o3d_pc(xyz: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    print('[_to_o3d_pc] color: ', pcd.points)
        
    if color is not None:
        if len(color) != len(xyz): 
            color = np.array(color)[None].repeat(len(xyz), axis=0)
        pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd

def remove_pcd_outlier(pcd):
    voxel_down_pcd = pcd.voxel_down_sample(voxel_size=0.02)
    cl, ind = voxel_down_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_cloud = voxel_down_pcd.select_by_index(ind)
    
    return inlier_cloud


def interpolate_feature_dr(rast, pos, tris, feat):
    out, _ = dr.interpolate(feat[None], rast, tris)
    # out = dr.antialias(out, rast, pos[None], tris)
    out = out.detach().cpu().numpy()[:, ::-1, :, :]
    return out


def render_mesh_uv(
    glctx,
    mesh_fp,
    pattern_fp, 
    out_dir='', 
    render_scale=1.0,
    render_uv=True,           # render 2D relative coordinate
    render_seg=False           # render pattern semantics
    ):
    
    # load 3D mesh
    mesh_obj = read_obj(mesh_fp)
    
    # load 2D panel
    with open(pattern_fp, 'r', encoding='utf-8') as f: pattern_data = json.load(f)
    panel_data = dict([(x['id'], x) for x in pattern_data['panels']])
        
    verts = mesh_obj.points
    uv = mesh_obj.point_data['obj:vt']
        
    # verts = (verts - _GLOBAL_OFFSET[None]) / _GLOBAL_SCALE
        
    ####### Normalize vertex positions to zero mean && unit variance #######
    global_offset = _AVATAR_BBOX[0] + (_AVATAR_BBOX[1] - _AVATAR_BBOX[0]) / 2
    global_scale = _AVATAR_BBOX[1] - _AVATAR_BBOX[0]
    verts = (verts - global_offset[None]) / (global_scale[None] * 0.5)

    # verts_min = verts.min(axis=0)
    # verts_max = verts.max(axis=0)
    # global_offset = verts_min + (verts_max - verts_min) / 2
    # global_scale = max(verts_max - verts_min)
    # verts = (verts - global_offset[None]) / (global_scale * 0.5)
        
    # verts = (verts - verts.min(dim=0)[0]) / (verts.max(dim=0)[0] - verts.min(dim=0)[0] + 1e-6)
    # verts = (verts - _GLOBAL_MEAN[None]) / _GLOBAL_STD[None]
    ########################################################################
        
    verts = torch.from_numpy(verts).to(torch.float32).to('cuda')
    uv = torch.from_numpy(uv).to(torch.float32).to('cuda')
    
    # Normalized uv && vertex position (local coordinate of each panel)
    if render_uv: uv_local = uv.clone()
    # verts_local = verts.clone()

    bbox_2d = torch.cat([
        torch.min(uv[:, :2], axis=0, keepdim=True)[0] - 10.0, 
        torch.max(uv[:, :2], axis=0, keepdim=True)[0] + 10.0
        ])
    
    img_wh = bbox_2d[1] - bbox_2d[0]
    
    panel_ids = []
    tris = []                           # all triangles
    if render_seg: 
        tri_semantics, tri_coords = [], []   # triangle semantics
        
    for idx, panel_id in enumerate(mesh_obj.field_data['obj:group_tags']):
        
        if panel_id not in panel_data: continue
        
        panel_faces = mesh_obj.cells[idx].data
        vert_ids = np.unique(panel_faces)        
        
        ################### Transfer to local coordinate of current panel ###################
        # panel_bbox = torch.cat([
            
        #     torch.min(verts_local[vert_ids, :], dim=0, keepdim=True)[0], 
        #     torch.max(verts_local[vert_ids, :], dim=0, keepdim=True)[0]
        #     ])
        # verts_local[vert_ids, :] = (verts_local[vert_ids, :] - panel_bbox[0]) / (panel_bbox[1] - panel_bbox[0]) 
        #####################################################################################
        

        tris.append(torch.from_numpy(panel_faces).to('cuda'))
        panel_ids.append(torch.ones_like(tris[-1][:, 0], dtype=torch.int32, device='cuda') * idx)
        
        panel_center = torch.tensor(panel_data[panel_id]['center'], device='cuda')
        if render_seg:
            panel_seg_id = panel_data[panel_id]['label'].strip()
            panel_seg_id = _PANEL_CLS.index(panel_seg_id) + 1 if panel_seg_id in _PANEL_CLS else 0.
            tri_semantics.append(
                torch.ones_like(tris[-1][:, 0], dtype=torch.int32, device='cuda') * panel_seg_id)
        
        if render_uv:
            print('*** panel_uv: ', uv_local[vert_ids, :].shape, panel_center.shape)
            panel_uv = uv_local[vert_ids, :] - panel_center[None]
            panel_bbox = (
                torch.min(panel_uv, dim=0, keepdim=True)[0],
                torch.max(panel_uv, dim=0, keepdim=True)[0]
            )
            
            panel_uv = panel_uv / ((panel_bbox[1] - panel_bbox[0]) * 0.5 + 1e-6)      # normalize to [-1, 1]
            uv_local[vert_ids, :] = panel_uv
                
    panel_ids = torch.cat(panel_ids, dim=0)
    tris = torch.cat(tris, dim=0)
     
    pos = (uv[:, :2] - bbox_2d[0, :] + 1e-6) / (bbox_2d[1, :] - bbox_2d[0, :] + 1e-6) * 2.0 - 1.0
    pos = torch.cat([
        pos, 
        torch.zeros_like(pos[:, :1], dtype=torch.
                         float32, device='cuda'),
        torch.ones_like(pos[:, :1], dtype=torch.float32, device='cuda')
        ], dim=1)
     
    render_scale = min(render_scale, 2048.0 / max(img_wh))
    img_wh = (img_wh * render_scale / 8).ceil() * 8 

    rast, _ = dr.rasterize(
        glctx, pos[None], tris, resolution=[int(img_wh[1]), int(img_wh[0])])
    
    render_mask = (rast[..., -1:] > 0).float().detach().cpu().numpy()
    render_mask = render_mask[:, ::-1, :, :]
    
    out = interpolate_feature_dr(rast, pos, tris, verts)
    out_uv = interpolate_feature_dr(rast, pos, tris, uv_local) if render_uv else None
        
    out_seg, out_tri_index = None, None
    if render_seg:
        tri_semantics = torch.cat([torch.zeros_like(tri_semantics[0][:1])] + tri_semantics, dim=0).long().detach().cpu().numpy()
        tri_index = rast[..., 3][0].long().detach().cpu().numpy()[::-1, :]
        out_seg = tri_semantics[tri_index]
        out_tri_index = tri_index
        
    if out_dir:
        # np.savez(os.path.join(out_dir, 'data.npz'), geo=out[0], uv=out_uv[0] if render_uv else None, seg=out_seg if render_seg else None)
        # np.savez(os.path.join(out_dir, 'geo.npz'), geo=out[0])
        # np.save(os.path.join(out_dir, 'geo.npy'), out[0])                
        imageio.imsave(
            os.path.join(out_dir, 'geo.png'), 
            np.clip(np.rint((out[0] + 1.0) * 0.5 * 255) * render_mask[0], 0, 255).astype(np.uint8))
        if render_uv:
            # np.save(os.path.join(out_dir, 'uv.npy'), out_uv[0])
            uv_img = np.clip(np.rint((out_uv[0] + 1.0) * 0.5 * 255) * render_mask[0], 0, 255).astype(np.uint8)
            uv_img[..., -1] = 0
            imageio.imsave(
                os.path.join(out_dir, 'uv.png'), uv_img)
        if render_seg:
            # np.savez(os.path.join(out_dir, 'seg.npy'), out_seg)
            # np.save(os.path.join(out_dir, 'tri_index.npy'), out_tri_index)
            imageio.imsave(
                os.path.join(out_dir, 'seg.png'),
                np.clip(np.rint(_PANEL_COLORS[out_seg, :] * 255), 0, 255).astype(np.uint8)
            )

    result = {'geo': out[0], 'uv': out_uv[0, :, :, :2], 'seg': out_seg, 'tri_index': out_tri_index}

    return result


def process_item(data_idx, data_item, args, glctx):
    try:
        os.makedirs(args.output, exist_ok=True)
        output_fp = os.path.join(args.output, '%05d.pkl' % (data_idx))    
        
        img_output_dir = os.path.join(args.output, 'vis', '%05d' % (data_idx))
        os.makedirs(img_output_dir, exist_ok=True)
        
        mesh_fp = os.path.join(data_item, os.path.basename(data_item)+'.obj')
        pattern_fp = os.path.join(data_item, 'pattern.json')
        
        result = render_mesh_uv(
                glctx, 
                mesh_fp,
                pattern_fp,
                out_dir=img_output_dir,
                render_uv=args.render_uv, 
                render_seg=args.render_seg)
        result['data_fp'] = data_item
        
        print('Checking output...')
        for key in result.keys(): print(key, result[key].shape if type(result[key]) == np.ndarray else result[key])
        
        # with open(output_fp, 'wb') as f: pickle.dump(result, f)
        
        np.savez_compressed(
            output_fp.replace('pkl', 'npz'), 
            geo=result['geo'], 
            seg=result['seg'].astype(np.uint8),
            uv=result['uv'],
            tri=result['tri_index'].astype(int)
            )
        
        return True, data_item
                        
    except Exception as e:
        return False, f"{data_item} | [ERROR] {e}"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Draw panel bbox 3D")
    parser.add_argument("-i", "--input", default="0000", type=str, help="Input directory.")
    parser.add_argument("-o", "--output", default='0000', type=str, help="Output directory.")
    parser.add_argument("-r", "--range", default=None, type=str, help="Data range.")
    
    parser.add_argument("--render_uv", action='store_true', help="Render 2D UV image.")
    parser.add_argument("--render_seg", action='store_true', default=True, help="Render 2D UV image.")
    
    args, cfg_cmd = parser.parse_known_args()

    glctx = dr.RasterizeCudaContext()
    print('[DONE] Init renderer.')
    
    data_root_dirs = args.input.split(',')
    print('Input directories: ', data_root_dirs)
    
    data_items = []
    for idx, data_root in enumerate(data_root_dirs):
        cur_data_items = sorted([os.path.dirname(x) for x in glob(
            os.path.join(data_root, '**', 'pattern.json'), recursive=True)])
        data_items += cur_data_items
        print('[%02d/%02d] Found %d items in %s.'%(idx+1, len(data_root_dirs), len(cur_data_items), data_root))
    print('Total items: ', len(data_items))
    
    log_file = os.path.join(args.output, 'app.log')
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            processed = [x.split("\t")[0] for x in lines if x.split("\t")[1].strip() == "1"]
            data_items = [x for x in data_items if x not in processed]
        
    if args.range is not None:
        begin, end = args.range.split(",")
        begin, end = max(0, int(begin)), min(int(end), len(data_items))
        data_items = data_items[begin:end]
        print("Extracting range: %d %s" % (len(data_items), args.output))

    os.makedirs(args.output, exist_ok=True)


    failed_items = []
    with open(log_file, 'a+') as f:
        with ThreadPoolExecutor(max_workers=8) as executor:  # 可以调整max_workers以改变并行度
            futures = {executor.submit(
                process_item, data_idx, data_item, args, glctx): data_item \
                    for data_idx, data_item in enumerate(data_items[:8])}

            for future in tqdm(as_completed(futures), total=len(futures)):
                result, data_item = future.result()
                if not result:
                    data_item, err_code = data_item.split('|')
                    data_item = data_item.strip()
                    err_code = err_code.strip()
                    failed_items.append(data_item)
                    print('[ERROR] Failed to process data:', data_item, err_code)
                    f.write(f"{data_item}\t 0\n")
                else:
                    f.write(f"{data_item}\t 1\n")
                    
    with open(os.path.join(args.output, 'failed_items.log'), 'w') as f:
        for item in failed_items: f.write(item+'\n')
              
    # for data_idx, data_item in enumerate(tqdm(data_items)):
    #     try:
    #         if len(data_items) == 1: 
    #             output_dir = args.output
    #             os.makedirs(output_dir, exist_ok=True) 
    #         else: 
    #             output_dir = os.path.join(args.output, '%04d'%(data_idx)) 
    #             os.makedirs(output_dir, exist_ok=True)       
    #             with open(os.path.join(output_dir, 'info.txt'), 'w') as f: 
    #                 f.write('DATA_FP: \n\t%s'%(data_item))
                    
    #         if os.path.exists(os.path.join(output_dir, 'geo.png')): continue
                    
    #         mesh_fp = os.path.join(data_item, os.path.basename(data_item)+'.obj')

    #         pattern_fp = os.path.join(data_item, 'pattern.json')
    #         avatar_fp = os.path.join(data_item, 'avatar.json')
            
    #         if os.path.exists(avatar_fp):
    #             with open(avatar_fp, 'rb') as f: avatar_data = json.load(f)
    #             arrange_pts = np.array([x['xyz'] for x in avatar_data[0]['arrangements']], dtype=np.float32)
    #             global_bbox = [np.min(arrange_pts, axis=0), np.max(arrange_pts, axis=0)]
    #         else:
    #             global_bbox = _AVATAR_BBOX
                
    #         out, _, _ = render_mesh_uv(
    #             glctx, 
    #             mesh_fp,
    #             pattern_fp,
    #             output_dir,    
    #             render_uv=args.render_uv, 
    #             render_seg=args.render_seg)        
    #     except Exception as e:
    #         print('[FAILED] %s'%(data_item), str(e))
    #         failed_items.append(data_item)
            
    # if failed_items:
    #     with open(os.path.join(args.output, 'app.log'), 'w') as f:
    #         f.write('\n'.join(failed_items))