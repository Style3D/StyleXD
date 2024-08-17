import os
import sys
import json
import argparse
import re

from glob import glob
import numpy as np

# Makes core library available without extra installation steps
sys.path.insert(0, './external/')
sys.path.insert(1, './')
from external.pattern.rotation import euler_xyz_to_R

def sample_circle(start, end, circ_params, num_samples=32):
    from pygarment import CircleEdge
    radius, large_arc, right = circ_params
    edge = CircleEdge.from_points_radius(start, end, radius, large_arc, right)

    subedges = edge._subdivide([1.0/num_samples]*num_samples)
    verts = subedges.verts()
    verts = np.asarray(verts[::2] + verts[-1:])
    
    # rel coordinate to abs coordinate
    return verts


def sample_bezier(start, end, control_point, degree=2, num_samples=10):
    from geomdl import BSpline, utilities

    curve = BSpline.Curve()
    curve.degree = degree

    if degree == 2:
        curve.ctrlpts = [start.tolist(), control_point.tolist(), end.tolist()]
    else:
        curve.ctrlpts = [start.tolist(), control_point[0].tolist(), control_point[1].tolist(), end.tolist()]
    
    curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
    # curve.delta = 0.01
    curve.sample_size = num_samples
    evalpts = np.array(curve.evalpts)

    # print('*** check bezier: ', start, evalpts[0], end, evalpts[-1])

    # print('[Bezier] start: %s, end: %s, eval: %s'%(start, end, evalpts))
    return evalpts


def overlap(bbox1, bbox2):
    # bbox = [xmin, ymin, xmax, ymax]
    return not (
        bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3])


def get_edge_vec(pattern_spec, panel_id, edge_id):
    panel = pattern_spec["pattern"]["panels"][panel_id]
    verts = np.asarray(panel["vertices"])
    
    start = verts[panel["edges"][edge_id]["endpoints"][0]]
    end = verts[panel["edges"][edge_id]["endpoints"][1]]
    
    return end - start


def verts_to_style3d_coords(vertices, translation_2d):
    """Convert given vertices and panel (2D) translation to px coordinate frame & units"""
    # Put upper left corner of the bounding box at zero
    offset = np.min(vertices, axis=0)
    vertices = vertices - offset
    translation_2d = translation_2d + offset
    return vertices, translation_2d


def control_to_abs_coord(start, end, control_scale):
    """
    Derives absolute coordinates of Bezier control point given as an offset
    """
    edge = end - start
    edge_perp = np.array([-edge[1], edge[0]])

    control_start = start + control_scale[0] * edge
    control_point = control_start + control_scale[1] * edge_perp

    return control_point 


# return true if the curve is clockwise, else false
def check_winding_order(edge_seq, verts):
    ordered_verts = []
    for idx in range(1, len(edge_seq)):
        assert edge_seq[idx]["endpoints"][0] == edge_seq[idx-1]["endpoints"][1], "Edge sequence is not continuous!"

    verts = np.array(verts)
    ordered_verts = np.array([x['endpoints'][0] for x in edge_seq] + [0], dtype=int)
    ordered_verts = verts[ordered_verts]

    wind_checker =  np.sum(
        (ordered_verts[1:, 0] - ordered_verts[:-1, 0]) * \
        (ordered_verts[1:, 1] + ordered_verts[:-1, 1])
        )

    return wind_checker > 0

def check_winding_order1(edges):
    edgeIdx=[]
    verts=[]
    for idx,edge in enumerate(edges):
        edgeIdx.append({"endpoints":[idx,(idx+1)%len(edges)]})
        verts.append(edge["controlPoints"][0])
    return check_winding_order(edgeIdx,verts)



def reverse_2D_edge(edge):
    edge['endpoints'] = edge['endpoints'][::-1]
    if 'curvature' in edge:
        if edge['curvature']['type'] == 'circle':
            edge['curvature']['params'][2] = not edge['curvature']['params'][2]
        elif edge['curvature']['type'] == 'cubic' or edge['curvature']['type'] == 'quadratic':
            for ctrl_pt in edge['curvature']['params']:
                ctrl_pt[0] = 1.0 - ctrl_pt[0]
                ctrl_pt[1] = -ctrl_pt[1]
        else:
            raise ValueError('Unsupported curvature type: %s' % edge['curvature']['type'])
    
    return edge
    


def convert_edge_seq(key,edge_seq, verts, to_spline=False):
    for idx in range(1, len(edge_seq)):
        assert edge_seq[idx]["endpoints"][0] == edge_seq[idx-1]["endpoints"][1], \
            "Edge sequence is not continuous!"

    verts = np.array(verts)
    verts = verts - np.mean(verts, axis=0, keepdims=True)

    new_edge_seq = []

    for idx, edge in enumerate(edge_seq):
        start, end = verts[edge['endpoints'][0]], verts[edge['endpoints'][1]]
        
        if 'curvature' in edge:
            if isinstance(edge['curvature'], list) or edge['curvature']['type'] == 'quadratic':  
                control_scale = edge['curvature'] if isinstance(edge['curvature'], list) else \
                            edge['curvature']['params'][0]
                control_point = control_to_abs_coord(start, end, control_scale)
                
                if to_spline:
                    evalpts = sample_bezier(start, end, control_point, degree=3, num_samples=5)
                    new_evalpts = np.zeros((evalpts.shape[0], 3))
                    new_evalpts[:, :2] = evalpts
                    new_edge_seq.append({
                        "id": key+'_e%02d'%idx,
                        "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                        "controlPoints": new_evalpts.tolist()
                    })
                    
                else:
                    # convert to cubic bezier
                    ctrl_pt_1 = 2.0 / 3.0 * (control_point - end)
                    ctrl_pt_0 = 2.0 / 3.0 * (control_point - start)
                    new_edge_seq.append({
                        "id": key+'_e%02d'%idx,
                        "bezierPoints": [ctrl_pt_0.tolist() + [0.], ctrl_pt_1.tolist() + [0.]],
                        "controlPoints": [start.tolist() + [0.], end.tolist() + [0.]]
                    })


            elif edge['curvature']['type'] == 'circle':
                evalpts = sample_circle(start, end, edge['curvature']['params'], num_samples=32)

                new_evalpts = np.zeros((evalpts.shape[0], 3))
                new_evalpts[:, :2] = evalpts
                new_edge_seq.append({
                    "id": key+'_e%02d'%idx,
                    "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                    "controlPoints": new_evalpts.tolist()
                })
                
                
            elif edge['curvature']['type'] == 'cubic':
                control_point = np.array([control_to_abs_coord(start, end, p) for p in edge['curvature']['params']])

                if to_spline:
                    evalpts = sample_bezier(start, end, control_point, degree=3, num_samples=5)
                    new_evalpts = np.zeros((evalpts.shape[0], 3))
                    new_evalpts[:, :2] = evalpts
                    new_edge_seq.append({
                        "id": key+'_e%02d'%idx,
                        "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                        "controlPoints": new_evalpts.tolist()
                    })
                    
                else:
                    new_edge_seq.append({
                        "id": key+'_e%02d'%idx,
                        "bezierPoints": [(control_point[0]-start).tolist() + [0.], (control_point[1]-end).tolist() + [0.]],
                        "controlPoints": [start.tolist() + [0.], end.tolist() + [0.]]
                    })

        else:
            new_edge_seq.append({
                "id": key+'_e%02d'%idx,
                "bezierPoints": [[0, 0, 0], [0, 0, 0]],
                "controlPoints": [start.tolist() + [0.], end.tolist() + [0.]]
            })

    return new_edge_seq

    
# 将传入的panel_name进行匹配，返回对应的label
def panel_name_to_label(panel_name):
    label = None
    # 需要匹配的字符串列表
    # find_list = ["turtle","lapel","shoulder","sleeve","sl_","ftorso","btorso","wb","front","pant_f","back","pant_b","cuff_skirt"]
    # 匹配字典
    match_dict = {
        "turtle":"neck",
        "lapel":"neck",
        "shoulder":"shoulder",
        "sleeve":"arm",
        "sl_":"wrist",
        "ftorso":"bodyfront",
        "btorso":"bodyback",
        "wb":"waist",
        "front":"pelvisfront",
        "pant_f":"pelvisfront",
        "back":"pelvisback",
        "pant_b":"pelvisback",
        "skirt_f":"pelvisfront",
        "skirt_b":"pelvisback",
        "cuff_skirt":"foot"
        }

    # 遍历find_list中需要匹配的字符串片段
    for name_seg, match_label in match_dict.items():
        match = re.search(name_seg, panel_name) # 判断传入的panel_name中是否含有上述字符串片段
        if(match): # 如果包含
            label = match_label # 赋值label
            break
        else: # 如果不包含，就继续检测下一个
            continue

    return label


def get_panel_center(panel, is_cw: bool = False, max_panel_width=50):

    panel_center = np.asarray(panel['translation'][:2])
    verts_center = np.mean(np.asarray(panel['vertices']), axis=0)
    panel_center = panel_center + verts_center

    if is_cw: panel_center[0] = -panel_center[0]

    if panel['translation'][2] >= 0: panel_center[0] = panel_center[0] + max_panel_width
    else: panel_center[0] = panel_center[0] - max_panel_width

    return panel_center.tolist()


def get_panel_center_3d(panel, translation, offset=(0., 0.0, 0)):
    panel_center = np.asarray(translation)
    panel_rotation = np.asarray(panel["rotation"])
    rotation_matrix = euler_xyz_to_R(panel_rotation)
    verts_center = np.mean(np.asarray(panel['vertices']), axis=0)
    rotated_verts_center = np.array(rotation_matrix.dot(verts_center)).squeeze()
    panel_center = panel_center + rotated_verts_center
    
    panel_center = panel_center + np.array(offset)
    
    # special cases
    if 'pant' in panel['id']: panel_center[1] += 10.0
    if 'sleeve' in panel['id']: panel_center[0] *= 0.8

    return panel_center.tolist()


def flip_panel(panel):
    panel["vertices"] = [[-vert[0], vert[1], 0] for vert in panel["vertices"]]
    for i in range(len(panel["edges"])):
        panel["edges"][i]["bezierPoints"] = [[-point[0], point[1], point[2]] for point in panel["edges"][i]["bezierPoints"]]
        panel["edges"][i]["controlPoints"] = [[-point[0], point[1], point[2]] for point in panel["edges"][i]["controlPoints"]]
    return panel


def scale_panel(panel, scale: float):
    panel["vertices"] = [[scale*vert[0], scale*vert[1], scale*vert[2]] for vert in panel["vertices"]]
    for i in range(len(panel["edges"])):
        panel["edges"][i]["bezierPoints"] = [[scale*point[0], scale*point[1], scale*point[2]] for point in panel["edges"][i]["bezierPoints"]]
        panel["edges"][i]["controlPoints"] = [[scale*point[0], scale*point[1], scale*point[2]] for point in panel["edges"][i]["controlPoints"]]
    panel["center"] = (np.array(panel["center"], dtype=float) * scale).tolist()
    panel["translation"] = (np.array(panel["translation"], dtype=float) * scale).tolist()
    return panel


def convert_segment_to_stitch_segment(segment, is_first: bool):
    panel_id = segment["panel"]
    edge_id = segment["edge"]

    if is_first:
        return {
            "start": {
                "clothPieceId": panel_id,
                "edgeId": panel_id+'_e%02d'%edge_id,
                "param": 0
            },
            "end": {
                "clothPieceId": panel_id,
                "edgeId": panel_id+'_e%02d'%edge_id,
                "param": 1
            },
            "isCounterClockWise": False
        }
    else:
        return {
            "start": {
                "clothPieceId": panel_id,
                "edgeId": panel_id+'_e%02d'%edge_id,
                "param": 1
            },
            "end": {
                "clothPieceId": panel_id,
                "edgeId": panel_id+'_e%02d'%edge_id,
                "param": 0
            },
            "isCounterClockWise": True
        }


def convert_stitch(pattern_spec, stitch):
    
    _ = pattern_spec    # unused for now
    
    panel_id_0, edge_id_0 = stitch[0]["panel"], stitch[0]["edge"]
    panel_id_1, edge_id_1 = stitch[1]["panel"], stitch[1]["edge"]
    
    # edge_0 = get_edge_vec(pattern_spec, panel_id_0, edge_id_0)
    # edge_1 = get_edge_vec(pattern_spec, panel_id_1, edge_id_1)
    
    # flip = edge_0.dot(edge_1) > 0
    
    # print('*** panel_ids: ', panel_id_0, panel_id_1, panel_id_0.split('_tiered_'), panel_id_1.split('_tiered_'))
    flip = panel_id_0.split('_tiered_')[0] == panel_id_1.split('_tiered_')[0] and 'tiered' in panel_id_1
    
    # print('[Check Flip] %s-%s : %s'%(panel_id_0, panel_id_1, flip), edge_0, edge_1)
    stitch_results = [
        {
            "start": {
                "clothPieceId": panel_id_0,
                "edgeId": panel_id_0+'_e%02d'%edge_id_0,
                "param": 0
            },
            "end": {
                "clothPieceId": panel_id_0,
                "edgeId": panel_id_0+'_e%02d'%edge_id_0,
                "param": 1
            },
            "isCounterClockWise": False
        },
        {
            "start": {
                "clothPieceId": panel_id_1,
                "edgeId": panel_id_1+'_e%02d'%edge_id_1,
                "param": 1 if not flip else 0
            },
            "end": {
                "clothPieceId": panel_id_1,
                "edgeId": panel_id_1+'_e%02d'%edge_id_1,
                "param": 0 if not flip else 1
            },
            # TODO: False for tiered dress.
            "isCounterClockWise": True if not flip else False
        }
    ]
    
    return stitch_results


# Rearrange panels in 2D, such that they will not overlap
def rearrange_panels_2D(pattern_spec):
    raise NotImplementedError("TODO: Rearrange panels in 2D, such that they will not overlap.")


def convert_pattern(pattern_spec, offset=(0., 7.77, 0), use_edge_seq=True, rearrange_panels=False):
    
    # 获取原始的板片和缝纫线数据
    panels = pattern_spec["pattern"]["panels"]
    stitches = pattern_spec["pattern"]["stitches"]
    
    # 创建一个空的字典，用于存储目标的板片和缝纫线数据
    result = {}

    # 创建一个空的列表，用于存储目标的板片数据
    result["panels"] = []
    # 用于缝纫线特殊判断翻领
    panel_label_map = {}
    # 遍历原始的板片数据
    for key, panel_data in panels.items():
        # 创建一个空的字典，用于存储目标板片数据
        panel = {}
        # 设置板片的id，先用板片名称即key代替
        panel["id"] = key

        # 设置板片的label，调用panel_name_to_label()得到label
        panel["label"] = panel_name_to_label(key)
        panel_label_map[key] = True if re.search("lapel", key) else False

        # 设置板片的顶点，先用原始的顶点数据代替
        panel["vertices"] = [vert+[.0] for vert in panel_data["vertices"]]

        # 判断原始版片的方向
        is_cw = check_winding_order(panel_data["edges"], panel_data["vertices"])
        # print('[Check CCW] %20s : %s'%(key, is_cw), [x['endpoints'][0] for x in panel_data["edges"]])
        panel['winding'] = int(is_cw)

        # 转换板片的边
        panel["edges"] = convert_edge_seq(
            key,
            panel_data["edges"], panel_data["vertices"],
            to_spline=False)

        # 转换板片的bbox中心
        panel["center"] = get_panel_center(panel_data, is_cw)
        # print(f'{key} panel center: {panel["center"]}')

        # 3D旋转(角度制)
        panel["rotation"] = panel_data["rotation"]

        # 翻转
        if not (is_cw ^ (not panel_label_map[key])):
            panel = flip_panel(panel)
            panel["rotation"] = [
                panel_data["rotation"][0],
                panel_data["rotation"][1] + 180,
                panel_data["rotation"][2]
            ]

        # 转换版片的边，改成顺时针方向
        is_cw = check_winding_order1(panel["edges"])
        if not is_cw:
            panel["edges"] = panel["edges"][::-1]
            for edge in panel["edges"]:
                edge["bezierPoints"] = edge["bezierPoints"][::-1]
                edge["controlPoints"] = edge["controlPoints"][::-1]

        # 3D位置
        panel["translation"] = get_panel_center_3d(panel, panel_data["translation"], offset=offset)
        # 将vertex位置转换到版片中心
        panel['vertices'] = [edge['controlPoints'][0] for edge in panel['edges']]
        # 单位标准化
        panel = scale_panel(panel, 1000.0/pattern_spec["properties"]["units_in_meter"])
        # 将板片数据添加到列表中
        result["panels"].append(panel)

    # convert stitches
    result["stitches"] = []
    for stitch in stitches:
        result["stitches"].append(convert_stitch(pattern_spec, stitch))        

    if use_edge_seq:
        # convert to 2024.03 format
        for p_idx in range(len(result["panels"])):
            panel = result["panels"][p_idx]
            result["panels"][p_idx] = {
                "center": panel['center'],
                "id": panel['id'],
                "label": panel['label'],
                "seqEdges": [{
                    "type": 3,
                    "circleType": 0,
                    "edges": panel['edges'],
                    "vertices": panel['vertices'] + [panel['vertices'][0]],
                }],
                "rotation":panel["rotation"],
                "translation": panel["translation"],                    
                "particle_dist": 6.0
            }

    return result


def convert_file(input_fp, output_fp=''):
    try:
        with open(input_fp, 'r', encoding='utf-8') as f:
            pattern_spec = json.load(f)
            result_pattern = convert_pattern(pattern_spec)
            with open(output_fp, 'w', encoding='utf-8') as f:
                json.dump(result_pattern, f, indent=4)
        return True
    
    except Exception as e:
        print('[FAILED] processing %s: %s' % (input_fp, e))
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert GarmentCode spec to pattern.json")
    parser.add_argument("-i", "--input", type=str, help="Input spec path.")
    parser.add_argument("-o", "--output", type=str, help="Output directory.")
    parser.add_argument("-f", "--file", default="specification.json", type=str, help="Input file name.")
    
    args, cfg_cmd = parser.parse_known_args()

    input_files = glob(os.path.join(args.input, '**', args.file), recursive=True) if os.path.isdir(args.input) else [args.input]
    os.makedirs(args.output, exist_ok=True)

    print("[INFO] Found %d files in %s" % (len(input_files), args.input))
    
    succee_cnt = 0
    for input_file in input_files:
        succee_cnt += int(convert_file(
            input_file, 
            os.path.join(
                args.output, 
                os.path.basename(os.path.dirname(input_file))+".json"
            )))
        
    print('[DONE] processing %d files, %d succeed, %d failed' % (len(input_files), succee_cnt, len(input_files)-succee_cnt))