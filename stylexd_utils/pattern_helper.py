# -*- coding: utf-8 -*-
"""Helper functions for processing pattern.json used in the AIGP project.
"""

import json
import copy

import numpy as np

from matplotlib import pyplot as plt

from typing import List, Dict

from geomdl import fitting, BSpline, utilities

from scipy.spatial import distance_matrix

from .obj_helper import read_obj


def convert_label_cn_to_eng(label: str) -> str:
    label = label.split('|')[0].strip()
    if "帽" in label: return label.replace("帽", "head")
    elif "领" in label: return label.replace("领", "neck")
    elif "肩" in label: return label.replace("肩", "shoulder")
    elif "袖片" in label: return label.replace("袖片", "arm")
    elif "袖口" in label: return label.replace("袖口", "wrist")
    elif "衣身前中" in label: return label.replace("衣身前中", "bodyfront")
    elif "衣身后中" in label: return label.replace("衣身后中", "bodyback")
    elif "衣身侧" in label: return label.replace("衣身侧", "bodyside")
    elif "腰头" in label: return label.replace("腰头", "waist")
    elif "裙前中" in label: return label.replace("裙前中", "pelvisfront")
    elif "裙后中" in label: return label.replace("裙后中", "pelvisback")
    elif "裙侧" in label: return label.replace("裙侧", "pelvisside")
    else: 
        print('[CONVERT_LABEL_CN2ENG] Unknown label: ', label)
        return label


def triangle_area(v1, v2, v3):
    """Calculate the area of a triangle given its vertices."""
    return 0.5 * np.linalg.norm(np.cross(v2 - v1, v3 - v1))


def triangle_normal(v1, v2, v3):
    """Calculate the area and normal of a triangle given its vertices."""
    side1 = v2 - v1
    side2 = v3 - v1
    normal = np.cross(side1, side2)
    normalized_normal = normal / np.linalg.norm(normal) if np.linalg.norm(normal) != 0 else np.zeros(3)
    return normalized_normal


def center_of_mass_with_normal(V, F):
    """Calculate the center of mass and average normal of a triangular mesh."""
    centroid_sum = np.zeros(3)
    normal_sum = np.zeros(3)
    total_mass = 0

    for face in F:
        v1, v2, v3 = V[face[0]], V[face[1]], V[face[2]]
        centroid = (v1 + v2 + v3) / 3

        area = triangle_area(v1, v2, v3)
        normal = triangle_normal(v1, v2, v3)

        centroid_sum += centroid * area
        normal_sum += normal * area
        total_mass += area

    center_mass = centroid_sum / total_mass if total_mass > 0 else None
    average_normal = normal_sum / total_mass if total_mass > 0 else None
    
    return center_mass, average_normal


def extract_ref_points(panel_label, arrangements, verts, faces=None, normals=None, offset=0.1):

    # if len(panel_label.split("|")) > 1:
    #     p_conf = float(panel_label.split("|")[1])
    #     if p_conf < 0.5: panel_label = "unknown"
    #     else: panel_label = panel_label.split("|")[0]
    #     # print('*** p_conf: ', panel_label, p_conf) 

    panel_label = convert_label_cn_to_eng(panel_label)

    if panel_label.startswith('head'):
        ref_pts = [x for x in arrangements if "head" in x['name'].lower()]
    elif panel_label.startswith('neck'):
        return "neckBackCenter_0"
        # ref_pts = [x for x in arrangements if "neck" in x['name'].lower()]
    elif panel_label == 'shoulder':
        ref_pts = [x for x in arrangements if "shoulder" in x['name'].lower()]
    elif panel_label == 'arm':
        ref_pts = [x for x in arrangements if x['name'].lower().startswith("arm") \
                   and ('center' in x['name'].lower() or 'under' in x['name'].lower())]
        verts[:, 0] = verts[:, 0] * 1.05
    elif panel_label == 'wrist':
        ref_pts = [x for x in arrangements if x['name'].lower().startswith('wrist')]
    elif panel_label == 'bodyfront':
        ref_pts = [x for x in arrangements if x['name'].lower().startswith('bodyfront')]
        verts[:, 2] = verts[:, 2] + offset
    elif panel_label == 'bodyback':
        ref_pts = [x for x in arrangements if x['name'].lower().startswith('bodyback')]
        verts[:, 2] = verts[:, 2] - offset
    elif panel_label == 'bodyside':
        ref_pts = [x for x in arrangements if x['name'].lower().startswith('body')]
    elif panel_label == 'hem' or panel_label == 'waist':
        ref_pts = [x for x in arrangements if (
            x['name'].lower().startswith('skirt') or \
            x['name'].startswith('bodyBackCenter4') or \
            x['name'].startswith('bodyFrontCenter4') or \
            x['name'].startswith('bodyBackCenter5') or \
            x['name'].startswith('bodyFrontCenter5')
            )]
    elif panel_label == 'pelvisfront':
        ref_pts = [x for x in arrangements if (
            x['name'].startswith('bodyFrontCenter5') or \
            x['name'] == 'BodyLeft2' or \
            x['name'] == 'BodyRight2' or \
            x['name'].startswith('Skirt')
            )]
        verts[:, 2] = verts[:, 2] + offset
    elif panel_label == 'pelvisback':
        ref_pts = [x for x in arrangements if (
            x['name'].startswith('bodyFrontCenter5') or \
            x['name'] == 'BodyLeft2' or \
            x['name'] == 'BodyRight2' or \
            x['name'].startswith('Skirt')
            )]
        verts[:, 2] = verts[:, 2] - offset
    else:
        ref_pts = arrangements

    ref_xyz = np.stack([np.array(x['xyz']) for x in ref_pts])
    ref_label = [x['name'] for x in ref_pts]
        
    ############### Filtering out upper half verts ################
    # # up axis: "Y" -> verts[:, 1]
    # height_level = np.mean(verts[:, 1])
    # verts = verts[verts[:, 1] > height_level, :]
    # print('*** verts: ', verts.shape)
    ###############################################################
        
    # centroid distance
    # if faces is not None:
    #     print('*** verts: ', np.min(verts, axis=0), np.max(verts, axis=0), offset)
    #     mesh_centroid, mesh_normal = center_of_mass_with_normal(verts, faces)
    #     print('*** centroid dist: ', mesh_centroid, mesh_normal, offset)
    #     mesh_centroid = mesh_centroid + mesh_normal * offset
    #     verts = verts + mesh_normal * offset
    # elif normals is not None:
    #     print('*** norm mean: ', np.mean(normals, axis=0, keepdims=True))
    #     verts = verts + np.mean(normals, axis=0, keepdims=True) * offset
    #     mesh_centroid = np.mean(verts, axis=0, keepdims=True)
    # else:
    
    mesh_centroid = np.mean(verts, axis=0, keepdims=True)
    dist = np.linalg.norm(ref_xyz - mesh_centroid, axis=1)
    res = ref_label[np.argmin(dist)]

    # # pair-wise distance
    # dist = distance_matrix(verts, ref_xyz)
    # res = ref_label[np.argmin(np.mean(dist, axis=0))]

    return res


def _poly_area(vertices):
    if not isinstance(vertices, np.ndarray):
        vertices = np.asarray(vertices)

    x, y = vertices[:, 0], vertices[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _line_intersection(s1, e1, s2, e2):
    """
    Calculate the intersection point of two lines defined by points s1, e1 and s2, e2.

    Args:
    s1, e1: numpy arrays defining the start and end points of the first line.
    s2, e2: numpy arrays defining the start and end points of the second line.

    Returns:
    A tuple containing the intersection point, or None if the lines do not intersect.
    """
    # Line1 represented as a1x + b1y = c1
    a1 = e1[1] - s1[1]
    b1 = s1[0] - e1[0]
    c1 = a1*s1[0] + b1*s1[1]

    # Line2 represented as a2x + b2y = c2
    a2 = e2[1] - s2[1]
    b2 = s2[0] - e2[0]
    c2 = a2*s2[0] + b2*s2[1]

    determinant = a1*b2 - a2*b1

    if determinant == 0:
        # Lines are parallel, no intersection
        return None
    else:
        x = (b2*c1 - b1*c2) / determinant
        y = (a1*c2 - a2*c1) / determinant
        return (x, y)


def normalize_edge_bezier(edge_spec: Dict, degree: int = 3) -> Dict:
    assert degree == 2 or degree == 3, 'Only support quadratic or cubic bezier curves.'

    bezierPts = np.asarray(edge_spec['bezierPoints'], dtype=np.float32)[:, :2]
    ctrlPts = np.asarray(edge_spec['controlPoints'], dtype=np.float32)[:, :2]

    print('*** ctrlPts: ', ctrlPts)

    assert len(ctrlPts) >= 2, 'At least two control points are required, i.e. start and end point of the edge.'

    if np.any(bezierPts) and len(ctrlPts) == 2:
        if degree == 3 and not np.any(bezierPts[1]):
            # quadratic bezier to cubic bezier
            bezierPts[1] = 2.0 / 3.0 * (bezierPts[0] + ctrlPts[0] - ctrlPts[1])
            bezierPts[0] = 2.0 / 3.0 * bezierPts[0]
        elif degree == 2 and np.any(bezierPts[1]):
            # cubic bezier to quadratic bezier
            bezierPts[0] = _line_intersection(ctrlPts[0], ctrlPts[0]+bezierPts[0], ctrlPts[1], ctrlPts[1]+bezierPts[1])
            bezierPts[1] = np.zeros(2, dtype=np.float32)
    else:
        if ctrlPts.shape[0] <= 2: 
            bezierPts[1] = 1.0 / 3.0 * (ctrlPts[0] - ctrlPts[1])
            bezierPts[0] = 1.0 / 3.0 * (ctrlPts[1] - ctrlPts[0])
        else:    
            if degree == 3 and ctrlPts.shape[0] < 5:
                _curve = fitting.interpolate_curve(
                            ctrlPts.tolist(), degree=2 if ctrlPts.shape[0] < 5 else 3)
                _curve.sample_size = 6
                _curve.evaluate()
                ctrlPts = np.array(_curve.evalpts)

            curve = fitting.interpolate_curve(ctrlPts.tolist(), degree=degree)

            bezierPts[0] = np.asarray(curve.ctrlpts[1]) - np.asarray(curve.ctrlpts[0])
            bezierPts[1] = np.asarray(curve.ctrlpts[2]) - np.asarray(curve.ctrlpts[3])

            ctrlPts[0] = np.asarray(curve.ctrlpts[0])
            ctrlPts[1] = np.asarray(curve.ctrlpts[3])

    bezierPts = np.concatenate([bezierPts, np.zeros_like(bezierPts[:, :1])], axis=-1)
    ctrlPts = np.concatenate([ctrlPts, np.zeros_like(ctrlPts[:, :1])], axis=-1)

    edge_spec = {
        'bezierPoints': bezierPts.tolist(),
        'controlPoints': ctrlPts.tolist(),
        "id": edge_spec["id"],
        "label": edge_spec["label"]
    }

    return edge_spec


def normalize_edge_bspline(edge_spec: Dict, max_num_ctrl_pts: int = 6) -> Dict:
    
    bezierPts = np.asarray(edge_spec['bezierPoints'])[:, :2]
    ctrlPts = np.asarray(edge_spec['controlPoints'])[:, :2]

    assert len(ctrlPts) >= 2, 'At least two control points are required.'

    if np.any(bezierPts) and len(ctrlPts) == 2:
        if not np.any(bezierPts[1]):
            # print('quadratic bezier')
            bezierPts[1] = 2.0 / 3.0 * (bezierPts[0] + ctrlPts[0] - ctrlPts[1])
            bezierPts[0] = 2.0 / 3.0 * bezierPts[0]
            
        _bezierPts = ctrlPts + bezierPts
        curve = BSpline.Curve()
        curve.degree = 3
        curve.ctrlpts = [ctrlPts[0].tolist()] + _bezierPts.tolist() + [ctrlPts[1].tolist()]
        curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))

        curve.sample_size = max_num_ctrl_pts
        curve.evaluate()
        ctrlPts = np.array(curve.evalpts)
    else:
        if ctrlPts.shape[0] <= 2:
            _t = np.linspace(0, 1, max_num_ctrl_pts).reshape(-1, 1)
            ctrlPts = (1 - _t) * ctrlPts[0] + _t * ctrlPts[1]

        elif ctrlPts.shape[0] != max_num_ctrl_pts:
            _curve = fitting.interpolate_curve(
                ctrlPts.tolist(), degree=2 if ctrlPts.shape[0] < 5 else 3)
            _curve.sample_size = max_num_ctrl_pts
            _curve.evaluate()
            ctrlPts = np.array(_curve.evalpts)

    bezierPts = np.zeros((2, 3))
    ctrlPts = np.concatenate([ctrlPts, np.zeros_like(ctrlPts[:, :1])], axis=-1)

    edge_spec = {
        'bezierPoints': bezierPts.tolist(),
        'controlPoints': ctrlPts.tolist(),
        "id": edge_spec["id"],
        "label": edge_spec["label"]
    }

    return edge_spec


def convert_edge_abs_to_rel(edge_spec: Dict, verts: np.ndarray):

    edge_crtl_pts = np.asarray(edge_spec['controlPoints'])[:, :2]
    edge_bezier_pts = np.asarray(edge_spec['bezierPoints'])
    if np.any(edge_bezier_pts):
        edge_bezier_pts = edge_bezier_pts[:, :2] + edge_crtl_pts    # convert from vector repr. to point repr.

    # Convert ctrl points to relative coordinate
    start = edge_crtl_pts[0, :]
    end = edge_crtl_pts[-1, :] 

    start_idx = np.argmin(np.linalg.norm(verts[:, :2] - start[None], axis=1))
    end_idx = np.argmin(np.linalg.norm(verts[:, :2] - end[None], axis=1))
    edge_spec["endPoints"] = [int(start_idx), int(end_idx)]

    # convert edge parameters    
    edge = end - start
    edge_len = np.linalg.norm(edge)

    def project_point(pt):
        converted = [None, None]
        point_vec = pt - start
        projected_len = edge.dot(point_vec) / edge_len
        converted[0] = projected_len / edge_len

        control_projected = edge * converted[0]
        vert_comp = point_vec - control_projected
        converted[1] = np.linalg.norm(vert_comp) / edge_len
        converted[1] *= -np.sign(np.cross(point_vec, edge)) 

        return converted

    converted_ctrl_pts = np.array([project_point(x) for x in edge_crtl_pts], dtype=np.float32)
    converted_ctrl_pts = np.concatenate([converted_ctrl_pts, np.zeros_like(converted_ctrl_pts[:, :1])], axis=-1)
    edge_spec['controlPoints'] = converted_ctrl_pts.tolist()

    if np.any(edge_bezier_pts):
        converted_bezier_pts = np.array([project_point(x) for x in edge_bezier_pts], dtype=np.float32)
        converted_bezier_pts = np.concatenate([converted_bezier_pts, np.zeros_like(converted_bezier_pts[:, :1])], axis=-1)
        edge_spec['bezierPoints'] = converted_bezier_pts.tolist()

    # print('[convert_edge_abs_to_rel] edge_spec: \n', edge_spec)
    return edge_spec


def convert_edge_rel_to_abs(edge_spec: Dict, verts: np.ndarray):

    start = verts[edge_spec["endPoints"][0], :2]
    end = verts[edge_spec["endPoints"][1], :2]

    edge = end - start                      
    edge_perp = np.array([-edge[1], edge[0]])    # Y-axis, perpendicular to edge

    def unproject_point(pt):
        conv_start = start + pt[0] * edge
        conv_point = conv_start + pt[1] * edge_perp
        return conv_point

    edge_crtl_pts = np.array(edge_spec['controlPoints'])[:, :2]
    edge_bezier_pts = np.array(edge_spec['bezierPoints'])[:, :2]

    converted_ctrl_pts = np.array([unproject_point(x) for x in edge_crtl_pts], dtype=np.float32)
    converted_ctrl_pts = np.concatenate([converted_ctrl_pts, np.zeros_like(converted_ctrl_pts[:, :1])], axis=-1)
    edge_spec['controlPoints'] = converted_ctrl_pts.tolist()

    if np.any(edge_bezier_pts):
        converted_bezier_pts = np.array([unproject_point(x) for x in edge_bezier_pts], dtype=np.float32)
        converted_bezier_pts = np.concatenate([converted_bezier_pts, np.zeros_like(converted_bezier_pts[:, :1])], axis=-1)
        edge_spec['bezierPoints'] = (converted_bezier_pts-converted_ctrl_pts).tolist()

    del edge_spec["endPoints"]

    return edge_spec


def modify_pattern_edge(
        pattern_spec: Dict, 
        output_dir: str = '',
        edge_label: str = "底摆线", 
        aug_dir: List = [0., -1., 0.], 
        step_size = 10, 
        num_steps = 60):

    output_specs = []

    for idx in range(num_steps):
        
        for panel in pattern_spec['panels']:
            rel_verts = []
            for edge in panel['edges']:
                if not "endPoints" in edge: 
                    edge = convert_edge_abs_to_rel(edge, np.array(panel['vertices']))
                if edge["label"] == edge_label:
                    rel_verts.append(edge["endPoints"][0])
                    rel_verts.append(edge["endPoints"][1])

            for vert_idx in rel_verts:
                panel['vertices'][vert_idx][0] += aug_dir[0] * step_size
                panel['vertices'][vert_idx][1] += aug_dir[1] * step_size
                panel['vertices'][vert_idx][2] += aug_dir[2] * step_size
        
        new_pattern_spec = copy.deepcopy(pattern_spec)
        for new_panel in new_pattern_spec['panels']:
            new_panel_verts = np.array(new_panel['vertices'])
            for new_edge in new_panel['edges']:
                new_edge = convert_edge_rel_to_abs(new_edge, new_panel_verts)

        output_specs.append(new_pattern_spec)

        if output_dir:
            with open(os.path.join(output_dir, 'panel_%04d.json'%(idx)), 'w') as f: 
                json.dump(new_pattern_spec, f, indent=4, ensure_ascii=False)
    
    return output_specs
            


def pattern_to_image(data_item, out_dir='', cmap=None):
    out_item_fp = os.path.join(out_dir, os.path.basename(data_item)).replace('.json', '.png')
    if os.path.exists(out_item_fp):
        return

    from geomdl import fitting, BSpline, utilities

    with open(data_item, 'r') as f: panel_data = json.load(f)

    # bounding box
    x_min, y_min = 999999, 999999
    x_max, y_max = -999999, -999999
    bbox_padding = 10

    panel_bboxes = {}

    fig, axs = plt.subplots(1, 1)

    for panel in panel_data['panels']:
        origin = np.array(panel['center'])[:2]
        
        poly_pts = []
        for edge in panel['edges']:
            # print('\t ', edge['id'])
            bezierPts = np.asarray(edge['bezierPoints'])[:, :2]
            ctrlPts = np.asarray(edge['controlPoints'])[:, :2]

            if np.any(bezierPts) and len(ctrlPts) == 2:
                if not np.any(bezierPts[1]):
                    # print('quadratic bezier')
                    bezierPts[1] = 2.0 / 3.0 * (bezierPts[0] + ctrlPts[0] - ctrlPts[1])
                    bezierPts[0] = 2.0 / 3.0 * bezierPts[0]
                    
                bezierPts = ctrlPts + bezierPts
                curve = BSpline.Curve()
                curve.degree = 3
                curve.ctrlpts = [ctrlPts[0].tolist()] + bezierPts.tolist() + [ctrlPts[1].tolist()]
                curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
                curve.sample_size = 100
                curve.evaluate()
                
                evalpts = np.array(curve.evalpts)

            else:            
                if ctrlPts.shape[0] <= 2: 
                    evalpts = ctrlPts
                else:    
                    curve = fitting.interpolate_curve(
                        ctrlPts.tolist(), degree=2 if ctrlPts.shape[0] < 5 else 3)
                    curve.sample_size = 100
                    curve.evaluate()
                    evalpts = np.array(curve.evalpts)
        
            poly_pts.append(evalpts)
            # plt.scatter(evalpts[:, 0], evalpts[:, 1], c=np.random.rand(3))


        poly_pts = np.concatenate(poly_pts, axis=0) + origin[None]
        
        fill_color = cmap[panel['label']] if cmap is not None else '#696969'
        axs.fill(poly_pts[:, 0], poly_pts[:, 1], facecolor=fill_color)

        _x_min, _y_min = np.array(poly_pts).min(axis=0)
        _x_max, _y_max = np.array(poly_pts).max(axis=0)
        panel_bboxes[panel['id']] = np.asarray((_x_min, _y_min, _x_max, _y_max))

        x_min, y_min = min(x_min, _x_min), min(y_min, _y_min)
        x_max, y_max = max(x_max, _x_max), max(y_max, _y_max)


    # bounding box
    x_min, y_min = x_min - bbox_padding, y_min - bbox_padding
    x_max, y_max = x_max + bbox_padding, y_max + bbox_padding

    for uuid in panel_bboxes:
        # matplotlib coordinate (x -> width, y -> height)
        panel_bboxes[uuid] = ((panel_bboxes[uuid].reshape(2, 2) - np.array([[x_min, y_min]])) / \
             (np.array([[x_max, y_max]]) - np.array([[x_min, y_min]]) + 1e-8)).reshape(-1)
        
        # transfer to image coordinate, s.t. panel = img[pb[0]:pb[2], pb[1]:pb[3], :]
        panel_bboxes[uuid] = np.array([
            1.0 - panel_bboxes[uuid][3],    # top-left corner, image coordinate
            panel_bboxes[uuid][0],   
            1.0 - panel_bboxes[uuid][1],    # bottom-right corner, image coordinate 
            panel_bboxes[uuid][2],         
        ])

    axs.set_aspect('equal', 'box')
    axs.set_xlim(x_min, x_max)
    axs.set_ylim(y_min, y_max)
    axs.axis("off")
    fig.tight_layout()

    if out_dir:
        out_item_fp = os.path.join(out_dir, os.path.basename(data_item)).replace('.json', '.png')
        plt.savefig(
            out_item_fp, 
            bbox_inches="tight", 
            pad_inches=0, 
            transparent=False,
            dpi=300)
    else:
        out_item_fp = None
        plt.show()
    
    plt.clf()
    plt.close()

    return out_item_fp, panel_bboxes



if __name__ == "__main__":
    import os
    import time
    import json
    from glob import glob
    import random
    from tqdm import tqdm
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Test pattern processing lib."
    )
    parser.add_argument(
        "-i", "--input", default=".\examples", type=str, help="Input directory including patterns and (optional) mesh objs."
    )
    parser.add_argument(
        "-o", "--output", default=".\outputs", type=str, help="Output directory, output pattern.json with lables, arrangements and stitches."
    )

    args, cfg_cmd = parser.parse_known_args()

    all_items = glob(f'{args.input}/*/pattern.json', recursive=True)
    print('Total number of items: ', len(all_items))

    os.makedirs(os.path.join(args.output, 'failed'), exist_ok=True)
    print('Creating output directory: ', args.output)

    succeed_cnt = 0

    for data_item in tqdm(sorted(all_items)):
        try:
            out_item_fp = os.path.join(args.output, os.path.basename(os.path.dirname(data_item)) + '.json')
            os.makedirs(os.path.dirname(out_item_fp), exist_ok=True)

            obj_id = os.path.basename(os.path.dirname(data_item))
            obj_fp = os.path.join(os.path.dirname(data_item), f'{obj_id}.obj')

            mesh_obj = read_obj(obj_fp)
            avatar_fp = os.path.join(os.path.dirname(data_item), 'avatar.json')
            with open(avatar_fp, 'r') as f: avatar_data = json.load(f)        

            pattern_to_image(data_item, out_item_fp, mesh_obj=mesh_obj, avatar=avatar_data)

            succeed_cnt += 1
        
        except Exception as e:
            print('Error: ', data_item, e)
            shutil.copyfile(data_item, os.path.join(args.output, 'failed', os.path.basename(data_item)))
            continue

    print('[DONE] processing pattern.json: ', succeed_cnt, len(all_items))
