import numpy as np
import glob
import os
import datetime
from collections import defaultdict
from tqdm import tqdm

import json
from matplotlib import pyplot as plt


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


def bspline_to_bezier(data_item, out_dir='', vis=False):
    """ Converting bspline curve to bezier curve.
        Args:
            data_item: input pattern.json (containing "panels" and "stitches").
    """
    from geomdl import fitting, BSpline, utilities

    with open(data_item, 'r') as f: panel_data = json.load(f)

    for panel in panel_data['panels']:
        # print(panel['id'], panel['label'], len(panel['vertices']), panel['center'])

        origin = np.array(panel['center'])
        poly_pts = []

        for edge in panel['edges']:
            # print('\t ', edge['id'])
            bezierPts = np.asarray(edge['bezierPoints'])[:, :2]
            ctrlPts = np.asarray(edge['controlPoints'])[:, :2]

            assert len(ctrlPts) >= 2, f"Number of control points should be more than 2, got {len(ctrlPts)}."
            assert len(bezierPts) == 2, f"Number of bezier points should be 2, got {len(bezierPts)}."

            if np.any(bezierPts) and len(ctrlPts) == 2:
                # convert quadratic bezier to cubic bezier
                if not np.any(bezierPts[1]):
                    bezierPts[1] = 2.0 / 3.0 * (bezierPts[0] + ctrlPts[0] - ctrlPts[1])
                    bezierPts[0] = 2.0 / 3.0 * bezierPts[0]
                    
                bezierPts = ctrlPts + bezierPts
                curve = BSpline.Curve()
                curve.degree = 3
                curve.ctrlpts = [ctrlPts[0].tolist()] + bezierPts.tolist() + [ctrlPts[1].tolist()]
                curve.knotvector = utilities.generate_knot_vector(curve.degree, len(curve.ctrlpts))
                curve.delta = 0.01
                evalpts = np.array(curve.evalpts)

            else:            
                if ctrlPts.shape[0] == 2: 
                    evalpts = ctrlPts
                    bezierPts = np.mean(ctrlPts, axis=0, keepdims=True)
                    # Straight Line
                    edge['bezierPoints'] = np.zeros((2, 3), dtype=np.float32).tolist()
                else:              
                    # BSpline to Cubic Bezier
                    _curve = fitting.interpolate_curve(ctrlPts.tolist(), degree=3)
                    _curve.delta = 0.2

                    curve = fitting.approximate_curve(_curve.evalpts, degree=3)
                    curve.delta = 0.01
                    evalpts = np.array(curve.evalpts)
                    bezierPts = np.array(curve.ctrlpts)

                    edge['bezierPoints'][0][:2] = (bezierPts[1]-bezierPts[0]).tolist()
                    edge['bezierPoints'][1][:2] = (bezierPts[2]-bezierPts[3]).tolist()
                    edge['bezierPoints'] = edge['bezierPoints'][:2]

                    edge['controlPoints'][0][:2] = bezierPts[0].tolist()
                    edge['controlPoints'][1][:2] = bezierPts[3].tolist()
                    edge['controlPoints'] = edge['controlPoints'][:2]

            if vis: poly_pts.append(evalpts)

        if vis: 
            poly_pts = np.concatenate(poly_pts, axis=0)
            plt.axis('off')
            plt.fill(poly_pts[:, 0] + origin[0], poly_pts[:, 1]+ origin[1], facecolor='#696969')


    if out_dir:
        out_item_fp = os.path.join(out_dir, os.path.basename(data_item))
        with open(out_item_fp, 'w') as f: 
            json.dump(panel_data, f)

    if vis:

        if out_dir:
            plt.savefig(
                out_item_fp.replace('.json', '.png'), 
                bbox_inches="tight", 
                pad_inches=0, 
                transparent=True, 
                dpi=300)

        else: plt.show()

        plt.clf()


def panel_edge_to_points(pattern_item: dict, delta = 0.01, cmap=None) -> dict:
    """ Covert panel (bezier/bspline) edge representation (with sewing) to point representation (w.o. stitch).
        Args:
            pattern_item: 
                input pattern.json (containing "panels" and "stitches"), please refer to 
                https://linctex.yuque.com/uz3v2g/project_doc/qiw5lb6f4lm67a1v#Mvh4G pattern.json 
                for more details.
        Returns:
            dict: converted panels.json including "uuid" and "points", please refer to 
                https://linctex.yuque.com/uz3v2g/project_doc/eycnm6q77y24vqv0 panels.json
                for more details.
    """
    from geomdl import fitting

    bbox = None

    with open(pattern_item, 'r') as f:
        pattern_data = json.load(f)

    all_verts = np.concatenate([
        np.array(panel['vertices']) + np.array(panel['center'])[None] for panel in pattern_data['panels']
    ])

    bbox = np.concatenate([
        np.min(all_verts[:, :2], axis=0, keepdims=True),
        np.max(all_verts[:, :2], axis=0, keepdims=True),
    ], axis=0)


    panels = []
    for panel in pattern_data['panels']:
        # sample points 
        origin = np.array(panel['center'])
        polyPts = []

        uuid = panel['id']

        for edge in panel['edges']:
            bezierPts = np.asarray(edge['bezierPoints'])[:, :2]
            ctrlPts = np.asarray(edge['controlPoints'])[:, :2]

            assert len(ctrlPts) == 2 and len(bezierPts) == 2, \
                f"Expect cubic bezier curve with 2 control points and 2 bezier points, got ({len(ctrlPts)}, {len(bezierPts)})."

            bezierPts = ctrlPts + bezierPts
            curve = fitting.interpolate_curve((ctrlPts[0], bezierPts[0], bezierPts[1], ctrlPts[1]), 3)
            curve.delta = delta

            evalPts = np.array(curve.evalpts)
            evalPts = ((evalPts + origin) - bbox[0:1]) / (bbox[1:] - bbox[:1])

            polyPts.append(evalPts)

        new_panel = {
            "uuid": uuid,
            "points": np.concatenate(polyPts, axis=0),
            "label": panel['label'] if 'label' in panel else 'none'
        }

        if cmap is not None:
            new_panel['color'] = cmap[new_panel['label']]

        panels.append(new_panel)

    return {"panels": panels, "bbox": bbox.reshape(-1).tolist()}


def studio_to_panel(studio_json: dict, panel_json: dict) -> dict:
    """change panel json content through
       studio json content

    Args:
        studio_json (dict): studio webhook: [ANNOTATION_CREATED, ANNOTATION_UPDATED]
        panel_json (dict): panel json content to be changed

    Returns:
        dict: changed panel json content
    """
    final_results = studio_json["annotation"]["result"][1:]
    origin_panels = panel_json["panels"]
    origin_bbox = panel_json["bbox"]

    new_panels = []

    for result in final_results:
        panel_id = result["id"]
        points = np.array(result["value"]["points"])
        label = result["value"]["polygonlabels"][0]

        for panel in origin_panels:
            if panel["uuid"] == panel_id:
                points = points / 100
                points[:, 1] = 1 - points[:, 1]
                points = points.tolist()

                panel["points"] = points
                panel["label"] = label

                new_panels.append(panel)

                break

    ret = {"panels": new_panels, "bbox": origin_bbox}

    return ret


def dataset_reader(dataset_f: str) -> list[str]:
    try:
        with open(dataset_f, "r") as f:
            data_urls = [x.strip() for x in f.readlines()]
        return data_urls
    except FileNotFoundError:
        print(f"[Error]The file '{dataset_f}' was not found.")
    except PermissionError:
        print(f"[Error]You don't have permission to read the file '{dataset_f}'.")
    except Exception as e:
        print(f"[Error]An error occurred: {e}")


def time_getter(timestamp=None):
    if not timestamp:
        timestamp = datetime.datetime.now().timestamp()
    dt_object = datetime.datetime.fromtimestamp(timestamp)
    formatted_date = dt_object.strftime("%Y-%m-%dT%H:%M:%S")
    return formatted_date


def string_to_timestamp(date_string):
    dt_object = datetime.datetime.strptime(date_string, "%Y-%m-%dT%H:%M:%S")
    return dt_object.timestamp()


def delete_old_labelling_vis(folder: str):
    """
    {folder} local posision include several set of images [x_0.png, x_1.png, x_2.png, x_3.png]
        with different timestamps
    remain the newest timestamp result
    """
    all_png_files = glob.glob(f"{folder}/*:*.png")

    if not all_png_files:
        return

    max_new_time = 0
    for file in all_png_files:
        a = file.split("/")[-1].split("_")[-2]
        time_stamp = string_to_timestamp(a)
        max_new_time = time_stamp if time_stamp > max_new_time else max_new_time

    newest = time_getter(max_new_time)

    to_delete = [x for x in all_png_files if newest not in x]

    if not to_delete:
        return

    for delete_f in to_delete:
        os.remove(delete_f)
        print(f"[INFO]Success delete old png {delete_f}.")


def split_dict(d: dict, n: int) -> list[dict]:
    if n <= 0:
        raise ValueError("[WARNING]Number of parts should be greater than 0.")

    keys = list(d.keys())
    avg_size = len(keys) // n
    parts = []

    for i in range(n):
        start_idx = i * avg_size
        if i == n - 1:  # For the last part, include any remaining items
            end_idx = len(keys)
        else:
            end_idx = start_idx + avg_size
        part = {k: d[k] for k in keys[start_idx:end_idx]}
        parts.append(part)

    return parts


def save_file(file_content: list, save_path: str) -> None:
    with open(save_path, "w") as f:
        for c in file_content:
            f.write(f"{c}\n")


def dataset_cutter(file_path: str, num_players: int) -> list[str]:
    if num_players == 1:
        return [file_path]

    data = dataset_reader(file_path)

    folders = defaultdict(list)
    for d in data:
        folder = d.split("/")[-2]
        folders[folder].append(d)

    final = split_dict(folders, num_players)

    # dict to list
    final_list = []
    for f_sub in final:
        f_sub_list = [item for sublist in f_sub.values() for item in sublist]
        final_list.append(f_sub_list)

    split_paths = []
    for i in tqdm(range(num_players)):
        save_path = file_path.replace(".txt", f"_{i}.txt")
        save_file(final_list[i], save_path)
        split_paths.append(save_path)

    return split_paths
