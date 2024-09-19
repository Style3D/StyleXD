# visualize point cloud
import argparse
import os

import numpy as np

import open3d as o3d
from matplotlib.colors import to_rgb

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

    "裤前中": {"alias": "pelvis front", "color": "#46B974"},
    "裤后中": {"alias": "pelvis back", "color": "#6B68F5"},
    "裤侧": {"alias": "pelvis side", "color": "#D37F50"},

    "橡筋": {"alias": "ruffles", "color": "#A8D4D2"},
    "木耳边": {"alias": "ruffles", "color": "#A8D4D2"},
    "袖笼拼条": {"alias": "ruffles", "color": "#A8D4D2"},
    "荷叶边": {"alias": "ruffles", "color": "#A8D4D2"},
    "绑带": {"alias": "ruffles", "color": "#A8D4D2"}
}

_CMAP = dict([(_CMAP[x]['alias'].replace(' ', ''), to_rgb(_CMAP[x]['color'])) for x in _CMAP])
print(_CMAP)


def _to_o3d_pc(xyz: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # print('[_to_o3d_pc] color: ', pcd.points)
    if color is not None:
        if len(color) != len(xyz) or len(xyz) == 3:
            color = np.array(color)[None].repeat(len(xyz), axis=0)
            if color.ndim == 1:
                color = np.array(color)
        pcd.colors = o3d.utility.Vector3dVector(color)

    return pcd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred_data_dir",
        required=True,
    )
    parser.add_argument(
        "--original_data_dir",
        required=True,
    )
    args = parser.parse_args()

    pred_data_dir = args.pred_data_dir
    original_data_dir = args.original_data_dir
    # pred_data_dir = "exp/StyleXD/garmentVisualization/pred_data_example/"
    # original_data_dir = "exp/StyleXD/garmentVisualization/original_data_example/"

    garment_list = os.listdir(pred_data_dir)
    for garment_name in garment_list:
        print(garment_name)

        # visualize predict results
        data_dir = os.path.join(pred_data_dir, garment_name)
        panel_points = []
        for panel_fp in os.listdir(data_dir):
            panel_cls = panel_fp.split('_')[0]
            panel_color = _CMAP[panel_cls]
            panel_pts = np.loadtxt(os.path.join(data_dir, panel_fp))
            if panel_pts.ndim==1:
                panel_pts = np.array(panel_pts).reshape(1,3)
            panel_points.append(_to_o3d_pc(panel_pts[:, :3], color=panel_color))
        o3d.visualization.draw_geometries(panel_points,window_name=f"{garment_name}_pred")
        input("press Enter key to continue")

        # visualize original garment
        data_dir = os.path.join(original_data_dir, garment_name)
        panel_points = []
        for panel_fp in os.listdir(data_dir):
            panel_cls = panel_fp.split('_')[0]
            panel_color = _CMAP[panel_cls]

            panel_pts = np.loadtxt(os.path.join(data_dir, panel_fp))
            panel_points.append(_to_o3d_pc(panel_pts[:, :3], color=panel_color))
        o3d.visualization.draw_geometries(panel_points,window_name=f"{garment_name}_original")
        input("press Enter key to continue")