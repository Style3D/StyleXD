from .obj_helper import Mesh, read_obj
import numpy as np

import imageio
import torch
import torch.nn.functional as F

import matplotlib
import matplotlib.pyplot as plt

from typing import Dict, Any, Tuple, Union

import nvdiffrast.torch as dr

import time


# Parsing color data
def __parsing_color_dict(profile_file):
    profile_data = json.load(profile_file)
    cmap = profile_data["pattern_cls"]

    for cls_key in cmap:
        cmap[cls_key] = np.array(matplotlib.colors.to_rgba(cmap[cls_key]["color"]))

    return cmap


def mapping_pattern_label(old_label):
    # right, left, center
    part_label = (
        old_label.replace("right", "")
        .replace("left", "")
        .replace("center", "")
        .replace("side", "")
    )
    dir_label_x = old_label.replace(part_label, "")

    # front, back
    tmp_part_label = part_label.replace("front", "").replace("back", "")
    dir_label_y = part_label.replace(tmp_part_label, "")

    if dir_label_y:
        new_label = tmp_part_label + dir_label_y
    elif tmp_part_label in ["ankle", "wrist", "shoulder", "arm", "waist"]:
        new_label = tmp_part_label
    elif dir_label_x in ["right", "left", "side"]:
        new_label = tmp_part_label + "side"
    else:
        new_label = part_label

    # print(f'Mapping Anno:\t{old_label} -> {new_label}')
    return new_label


# Functions from @Mateen Ulhaq and @karlo
def __set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    __set_axes_radius(ax, origin, radius)


def __set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def __look_at(eye, at, up):
    a = eye - at
    w = a / torch.linalg.norm(a)
    u = torch.cross(up, w)
    u = u / torch.linalg.norm(u)
    v = torch.cross(w, u)
    translate = torch.tensor(
        [[1, 0, 0, -eye[0]], [0, 1, 0, -eye[1]], [0, 0, 1, -eye[2]], [0, 0, 0, 1]],
        dtype=eye.dtype,
        device=eye.device,
    )
    rotate = torch.tensor(
        [
            [u[0], u[1], u[2], 0],
            [v[0], v[1], v[2], 0],
            [w[0], w[1], w[2], 0],
            [0, 0, 0, 1],
        ],
        dtype=eye.dtype,
        device=eye.device,
    )
    return rotate @ translate


def __perspective(fovy=0.7854, aspect=1.0, n=0.1, f=1000.0, device=None):
    y = np.tan(fovy / 2)
    return torch.tensor(
        [
            [1 / (y * aspect), 0, 0, 0],
            [0, 1 / -y, 0, 0],
            [0, 0, -(f + n) / (f - n), -(2 * f * n) / (f - n)],
            [0, 0, -1, 0],
        ],
        dtype=torch.float32,
        device=device,
    )


def __transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    # (x,y,z) -> (x,y,z,1)
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]


def __sample_rays(mtx, campos, reso=(512, 512), n=1.0, f=200.0):
    mtx = mtx.squeeze()
    campos = campos.squeeze()

    H, W = reso[0], reso[1]

    gy, gx = torch.meshgrid(
        torch.linspace(-1.0 + 1.0 / H, 1.0 - 1.0 / H, H, device=mtx.device),
        torch.linspace(-1.0 + 1.0 / W, 1.0 - 1.0 / W, W, device=mtx.device),
        indexing="ij",
    )

    imvp = torch.linalg.inv(mtx)
    gl_near = torch.stack((gx, gy, -torch.ones_like(gx), torch.ones_like(gx)), dim=0)

    world_near = torch.matmul(imvp, gl_near.view(4, -1)).view(4, H, W)
    world_near = (world_near / world_near[-1:, :, :])[:-1, :, :]

    ray_origin = campos[:, None, None].expand(-1, H, W)
    ray_dir = F.normalize(world_near - ray_origin, p=2, dim=0)

    ray_origin = ray_origin.permute(1, 2, 0)
    ray_dir = ray_dir.permute(1, 2, 0)

    tminmax = torch.cat(
        [
            n * torch.ones_like(ray_origin[..., -1:]),
            f * torch.ones_like(ray_origin[..., -1:]),
        ],
        dim=-1,
    )

    return ray_origin, ray_dir, tminmax


def __make_preview_cams(origin, axis, radius):
    pass


def render_anno_o3d(mesh: Mesh, annotation: Dict, cmap: Dict, output_fp=""):
    pass


def render_anno_plt(mesh: Mesh, annotation: Dict, cmap: Dict, output_fp=""):
    # tic = time.perf_counter()

    if cmap is None:
        cmap = matplotlib.cm.get_cmap("tab10")

    triangles = []
    tri_colors = []

    for cell_idx, cell in enumerate(mesh.cells):
        uuid = np.array(mesh.field_data["obj:group_tags"])[cell_idx]
        if uuid not in annotation:
            continue

        anno = annotation[uuid]

        triangles.append(cell.data)
        tri_colors.append(np.tile(cmap[anno], (len(cell.data), 1)))

    triangles = np.concatenate(triangles, axis=0)
    tri_colors = np.concatenate(tri_colors, axis=0)

    x, y, z = mesh.points[:, 0], mesh.points[:, 1], mesh.points[:, 2]

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection="3d")
    ax.set_axis_off()

    p3dc = ax.plot_trisurf(
        x, y, z, triangles=triangles, linewidth=0.0, antialiased=True
    )
    p3dc.set_fc(tri_colors)

    ax.set_box_aspect([1.0, 1.0, 1.0])
    __set_axes_equal(ax)

    ax.view_init(elev=0, azim=0, roll=0, vertical_axis="y")
    fig.savefig(
        output_fp.replace(".png", "_0.png"),
        format="png",
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    ax.view_init(elev=0, azim=90, roll=0, vertical_axis="y")
    fig.savefig(
        output_fp.replace(".png", "_1.png"),
        format="png",
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    ax.view_init(elev=0, azim=180, roll=0, vertical_axis="y")
    fig.savefig(
        output_fp.replace(".png", "_2.png"),
        format="png",
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    ax.view_init(elev=0, azim=270, roll=0, vertical_axis="y")
    fig.savefig(
        output_fp.replace(".png", "_3.png"),
        format="png",
        bbox_inches="tight",
        transparent=True,
        dpi=200,
    )

    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()

    # toc = time.perf_counter()
    # print('*** Render annotation: %.4f s' % (toc - tic))


def render_anno_dr(
    mesh: Mesh,
    annotation: Dict,
    cmap: Dict = {},
    output_fp="",
    glctx=None,
    cam_radius=2.0,
    mesh_scale=0.001,
    reso=800,
    anno_mapping_fn=None,
) -> list[str]:
    # tic = time.perf_counter()

    if anno_mapping_fn is not None:
        for key in annotation:
            annotation[key] = anno_mapping_fn(annotation[key])

    # build mesh
    triangles = []

    vert_colors = np.zeros((len(mesh.points), 4), dtype=np.float32) if cmap else None
    for cell_idx, cell in enumerate(mesh.cells):
        uuid = np.array(mesh.field_data["obj:group_tags"])[cell_idx]
        if uuid not in annotation:
            continue
        anno = annotation[uuid]
        triangles.append(cell.data)
        vert_colors[cell.data.reshape(-1)] = cmap[anno]
    triangles = np.concatenate(triangles, axis=0)

    # render
    if glctx is None:
        glctx = dr.RasterizeGLContext()

    output_list = []

    with torch.no_grad():
        vis_cams = (
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            * cam_radius
        )

        verts = torch.from_numpy(mesh.points).float().cuda() * mesh_scale
        tris = torch.from_numpy(triangles).int().cuda()
        if vert_colors is not None:
            vert_colors = torch.from_numpy(vert_colors).cuda()

        cam_target = torch.mean(verts, dim=0).float().cuda()
        cam_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")

        for i in range(len(vis_cams)):
            cam_pos = vis_cams[i] + cam_target
            mv = __look_at(cam_pos, cam_target, cam_up)
            proj = __perspective(device="cuda")
            mvp = proj @ mv

            # render loop
            pos_clip = __transform_pos(mvp, verts)

            rast_out, _ = dr.rasterize(glctx, pos_clip, tris, resolution=[reso, reso])
            seg_map, _ = dr.interpolate(vert_colors[None, ...], rast_out, tris)
            seg_map = dr.antialias(seg_map, rast_out, pos_clip, tris)

            seg_map = seg_map.squeeze().detach().cpu().numpy()

            file_name = output_fp.replace(".png", "_%d.png" % (i))
            imageio.imwrite(file_name, (seg_map * 255.0).astype(np.uint8))
            # print(f'[rerender]Rerender {file_name}')
            output_list.append(file_name)

    # toc = time.perf_counter()
    # print('*** Render annotation: %.4f s' % (toc - tic))

    return output_list


def render_seg_dr(
    mesh: Mesh,
    annotation: Dict,
    cmap: Dict = {},
    output_fp="",
    glctx=None,
    cam_radius=2.0,
    mesh_scale=0.001,
    reso=800,
    anno_mapping_fn=None,
    antialias=False,
    selector=None
) -> list[str]:
    # tic = time.perf_counter()

    if anno_mapping_fn is not None:
        for key in annotation:
            annotation[key] = anno_mapping_fn(annotation[key])

    # build mesh
    triangles = []
    tri_labels = []

    for cell_idx, cell in enumerate(mesh.cells):
        uuid = np.array(mesh.field_data["obj:group_tags"])[cell_idx]
        if uuid not in annotation: continue
        
        anno = annotation[uuid]
        
        if selector is not None and not selector(anno): continue
        
        triangles.append(cell.data)
        tri_labels.append(np.asarray(cmap[anno])[None].repeat(len(cell.data), axis=0))

    triangles = np.concatenate(triangles, axis=0)
    tri_labels = np.concatenate(tri_labels, axis=0)
    tri_labels = np.concatenate(
        [np.zeros_like(tri_labels[:1, ...]), tri_labels], axis=0
    )

    # print('*** tri_labels: ', triangles.shape, tri_labels.shape, tri_labels[0], tri_labels[1])

    # render
    if glctx is None:
        glctx = dr.RasterizeGLContext()

    output_list = []

    with torch.no_grad():
        vis_cams = (
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            * cam_radius
        )

        verts = torch.from_numpy(mesh.points).float().cuda() * mesh_scale
        tris = torch.from_numpy(triangles).int().cuda()
        tri_labels = torch.from_numpy(tri_labels).cuda()

        cam_target = torch.mean(verts, dim=0).float().cuda()
        cam_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")

        for i in range(len(vis_cams)):
            cam_pos = vis_cams[i] + cam_target
            mv = __look_at(cam_pos, cam_target, cam_up)
            proj = __perspective(device="cuda")
            mvp = proj @ mv

            # render loop
            pos_clip = __transform_pos(mvp, verts)

            rast_out, _ = dr.rasterize(glctx, pos_clip, tris, resolution=[reso, reso])
            seg_map = tri_labels[rast_out[..., 3].view(-1).long(), :].view(
                rast_out.shape[:3] + (tri_labels.shape[-1],)
            ).float()

            if antialias:
                print("*** seg_map: ", seg_map.shape, seg_map.dtype)
                seg_map = dr.antialias(seg_map, rast_out, pos_clip, tris)

            seg_map = seg_map.squeeze().detach().cpu().numpy()

            if output_fp.endswith(".png"):
                file_name = output_fp.replace(".png", "_%d.png" % (i))
                imageio.imwrite(file_name, (seg_map * 255.0).astype(np.uint8))
            elif output_fp.endswith(".npy"):
                file_name = output_fp.replace(".npy", "_%d.npy" % (i))
                with open(file_name, "wb") as f:
                    np.save(f, seg_map)

            output_list.append(file_name)

    # toc = time.perf_counter()
    # print('*** Render annotation: %.4f s' % (toc - tic))

    return output_list


def render_onehot_dr(
    mesh: Mesh,
    annotation: Dict,
    cmap: Dict = {},
    output_fp: str ="",
    glctx: Any = None,
    cam_radius: float = 2.0,
    mesh_scale: float = 0.001,
    reso: Union[int, Tuple[int, int]] = 800,
    anno_mapping_fn: Any = None,
    vis: bool = False,
    selector: Any = None
) -> list[str]:
    tic = time.perf_counter()

    if isinstance(reso, int): reso = (reso, reso)

    class_key_to_idx = dict([(x, idx) for (idx, x) in enumerate(cmap.keys())])

    if anno_mapping_fn is not None:
        for key in annotation:
            annotation[key] = anno_mapping_fn(annotation[key])

    # build mesh
    triangles = dict([(key, []) for key in cmap.keys()])

    for cell_idx, cell in enumerate(mesh.cells):
        uuid = np.array(mesh.field_data["obj:group_tags"])[cell_idx]
        if uuid not in annotation: continue
        
        anno = annotation[uuid]
        
        if selector is not None and not selector(anno): continue
        
        triangles[anno].append(cell.data)


    triangles = dict([(x, np.concatenate(triangles[x], axis=0)) for x in triangles.keys() if len(triangles[x])])

    # render
    if glctx is None:
        glctx = dr.RasterizeGLContext()

    output_list = []

    with torch.no_grad():
        vis_cams = (
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                ],
                dtype=torch.float32,
                device="cuda",
            )
            * cam_radius
        )

        verts = torch.from_numpy(mesh.points).float().cuda() * mesh_scale
        tris = dict([(x, torch.from_numpy(triangles[x]).int().cuda()) for x in triangles.keys()])

        cam_target = torch.mean(verts, dim=0).float().cuda()
        cam_up = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device="cuda")

        for i in range(len(vis_cams)):
            cam_pos = vis_cams[i] + cam_target
            mv = __look_at(cam_pos, cam_target, cam_up)
            proj = __perspective(device="cuda")
            mvp = proj @ mv

            # render loop
            pos_clip = __transform_pos(mvp, verts)

            res_onehot = np.zeros((reso[0], reso[1], len(class_key_to_idx)), dtype=bool)

            if vis:
                fig, axs = plt.subplots(nrows=2, ncols=np.ceil(len(tris.keys())/2).astype(int), figsize=(10, 5))
                fig.subplots_adjust(hspace = .001, wspace=.001)
                axs = axs.ravel()
                for ax in axs: ax.axis("off")

            for cls_idx, cls_key in enumerate(list(tris.keys())):
                rast_out, _ = dr.rasterize(glctx, pos_clip, tris[cls_key], resolution=reso)
                res_onehot[..., class_key_to_idx[cls_key]] = (rast_out[..., 3] > 0).squeeze().detach().cpu().numpy()
                
                if vis:
                    vis_img = res_onehot[..., class_key_to_idx[cls_key]].astype(float)
                    vis_img = vis_img[..., None].repeat(4, axis=-1) * cmap[cls_key].reshape(1, 1, 4)                    
                    axs[cls_idx].set_title(cls_key) 
                    axs[cls_idx].imshow(vis_img)

            output_list.append(res_onehot)
            if vis: plt.show()            

    output_list = np.stack(output_list, axis=0)

    if output_fp: np.savez(output_fp, output_list)

    toc = time.perf_counter()
    print('[DONE] Render annotation: %.4f s' % (toc - tic))

    return output_list


if __name__ == "__main__":
    import os
    import json
    from glob import glob
    import random
    from tqdm import tqdm
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Processing *.sproj -> cloth piece obj"
    )
    parser.add_argument(
        "-d", "--dir", default=".\dresses", type=str, help="Input directory."
    )
    parser.add_argument(
        "-n", "--num_items", default=-1, type=int, help="number of sampling items"
    )
    args, cfg_cmd = parser.parse_known_args()

    data_root = args.dir
    all_items = [
        os.path.dirname(x)
        for x in glob(os.path.join(data_root, "**", "*.obj"), recursive=True)
    ]
    if args.num_items > 0:
        all_items = random.choices(all_items, k=args.num_items)
    print(f"Processing {data_root}, {len(all_items)} in total (e.g. {all_items[0]})")

    # optional: load cmap
    tic = time.perf_counter()
    with open("/home/lry/code/garment_pattern_lib/info.json", "rb") as f:
        cmap = json.load(f)["pattern_cls"]
        for cls_key in cmap:
            cmap[cls_key] = np.array(matplotlib.colors.to_rgba(cmap[cls_key]["color"]))
    print("*** load cmap: ", time.perf_counter() - tic)

    for item_dir in tqdm(all_items):
        obj_fp = os.path.join(item_dir, os.path.basename(item_dir) + ".obj")
        panel_fp = os.path.join(item_dir, "panels.json")
        output_fp = os.path.join(item_dir, "labelling_vis.png")

        if not (os.path.exists(panel_fp) and os.path.exists(obj_fp)):
            continue

        # load annotations
        tic = time.perf_counter()
        with open(panel_fp, "rb") as f:
            data = json.load(f)
            annotations = dict(
                [(panel["uuid"], panel["label"]) for panel in data["panels"]]
            )
        # print("*** annotations: ", annotations, time.perf_counter() - tic)

        # load obj
        tic = time.perf_counter()
        mesh_obj = read_obj(obj_fp)
        # print('*** load mesh: num_verts=%d, num_panels=%d. [%.4f s]'%(
        #     len(mesh_obj.points), len(mesh_obj.cells), time.perf_counter() - tic))

        # render_anno_plt(mesh_obj, annotations, cmap, output_fp)
        glctx = dr.RasterizeGLContext()
        render_anno_dr(
            mesh_obj,
            annotations,
            cmap,
            output_fp,
            cam_radius=2.0,
            mesh_scale=0.001,
            reso=800,
            glctx=glctx,
            anno_mapping_fn=mapping_pattern_label,
        )

        if args.num_items > 0:
            shutil.copytree(
                item_dir,
                os.path.join(args.dir, "..", "test_data", os.path.basename(item_dir)),
            )
