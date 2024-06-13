import os
import numpy as np
import re
import json

import argparse

import matplotlib
# matplotlib.use('Agg')

import nvdiffrast.torch as dr

from utils.obj import read_obj
from glob import glob

from multiprocessing import Pool
import datetime

from utils.ding import markdown_msg, ding_msg, text_msg
from smd_helper import json2img
from utils import render as ru

import time


def _list_add(a, b, x, y):
    result = []
    for i in range(len(a)):
        result.append(x * a[i] + y * b[i])
    return result


# 得到小写的标签
def _extract_label(pt_str):
    res = re.search(r"[a-zA-Z]+", pt_str).end()
    return pt_str[:res].lower()


# 定义一个函数用于分类，传入name，返回字符串
def get_pattern_cls(name):
    # 新的分类
    categories = [
        "head",
        "neck",
        "shoulderleft",
        "shoulderright",
        "bodyfront",
        "bodyback",
        "bodyleft",
        "bodyright",
        "armleft",
        "armright",
        "wristleft",
        "wristright",
        "pelvisfront",
        "pelvisback",
        "pelvisleft",
        "pelvisright",
        "ankleleft",
        "ankleright",
    ]

    result = ""
    if name.startswith("ArrangePoint"):
        return result
    name_l = name.lower()
    for c in categories:
        if name_l.startswith(c):  # 转换为全小写再匹配
            result = c
            return result
    if result == "":
        print(name_l, " 没有对应的class")
    return result


def get_arrange_pts(arrDict):
    arrDictNew = {}
    # for i in range(1,6):
    # 头部

    if "headv1" in arrDict and "headv3" in arrDict:
        arrDictNew["head"] = 0.5 * arrDict["headv1"] + 0.5 * arrDict["headv3"]

    # 领子
    if "neckleft" in arrDict:
        arrDictNew["neckleft"] = arrDict["neckleft"]
    if "neckright" in arrDict:
        arrDictNew["neckright"] = arrDict["neckright"]
    if "neckfrontcenter_0" in arrDict:
        arrDictNew["neckfront"] = arrDict["neckfrontcenter_0"]
    if "neckfrontcenter_0" in arrDict:
        arrDictNew["neckback"] = arrDict["neckbackcenter_0"]

    # 衣身前（0，1，-1）
    # arrDictNew["bodyfrontcenter1_0"] = (arrDict["bodyfrontcenter1_-1"]+arrDict["bodyfrontcenter1_1"])/2
    arrDictNew["bodyfrontcenter1_0"] = (
        arrDict["bodyfrontcenter1_-1"] * 0.5 + arrDict["bodyfrontcenter1_1"] * 0.5
    )
    arrDictNew["bodyfrontcenter2_0"] = (
        arrDict["bodyfrontcenter2_-1"] * 0.5 + arrDict["bodyfrontcenter2_1"] * 0.5
    )
    arrDictNew["bodyfrontcenter3_0"] = (
        arrDict["bodyfrontcenter3_-1"] * 0.5 + arrDict["bodyfrontcenter3_1"] * 0.5
    )
    arrDictNew["bodyfrontcenter4_0"] = (
        arrDict["bodyfrontcenter4_-1"] * 0.5 + arrDict["bodyfrontcenter4_1"] * 0.5
    )
    arrDictNew["bodyfrontcenter5_0"] = (
        arrDict["bodyfrontcenter5_-1"] * 0.5 + arrDict["bodyfrontcenter5_1"] * 0.5
    )

    arrDictNew["bodyfrontcenter1_1"] = (
        0.5 * arrDictNew["bodyfrontcenter1_0"] + 0.5 * arrDict["bodyfrontcenter1_1"]
    )
    arrDictNew["bodyfrontcenter2_1"] = (
        0.5 * arrDictNew["bodyfrontcenter2_0"] + 0.5 * arrDict["bodyfrontcenter2_1"]
    )
    arrDictNew["bodyfrontcenter3_1"] = (
        0.5 * arrDictNew["bodyfrontcenter3_0"] + 0.5 * arrDict["bodyfrontcenter3_1"]
    )
    arrDictNew["bodyfrontcenter4_1"] = (
        0.5 * arrDictNew["bodyfrontcenter4_0"] + 0.5 * arrDict["bodyfrontcenter4_1"]
    )
    arrDictNew["bodyfrontcenter5_1"] = (
        0.5 * arrDictNew["bodyfrontcenter5_0"] + 0.5 * arrDict["bodyfrontcenter5_1"]
    )

    arrDictNew["bodyfrontcenter1_-1"] = (
        0.5 * arrDictNew["bodyfrontcenter1_0"] + 0.5 * arrDict["bodyfrontcenter1_-1"]
    )
    arrDictNew["bodyfrontcenter2_-1"] = (
        0.5 * arrDictNew["bodyfrontcenter2_0"] + 0.5 * arrDict["bodyfrontcenter2_-1"]
    )
    arrDictNew["bodyfrontcenter3_-1"] = (
        0.5 * arrDictNew["bodyfrontcenter3_0"] + 0.5 * arrDict["bodyfrontcenter3_-1"]
    )
    arrDictNew["bodyfrontcenter4_-1"] = (
        0.5 * arrDictNew["bodyfrontcenter4_0"] + 0.5 * arrDict["bodyfrontcenter4_-1"]
    )
    arrDictNew["bodyfrontcenter5_-1"] = (
        0.5 * arrDictNew["bodyfrontcenter5_0"] + 0.5 * arrDict["bodyfrontcenter5_-1"]
    )

    # 衣身后（0，1，-1）
    arrDictNew["bodybackcenter1_0"] = (
        0.5 * arrDict["bodybackcenter1_-1"] + 0.5 * arrDict["bodybackcenter1_1"]
    )
    arrDictNew["bodybackcenter2_0"] = (
        0.5 * arrDict["bodybackcenter2_-1"] + 0.5 * arrDict["bodybackcenter2_1"]
    )
    arrDictNew["bodybackcenter3_0"] = (
        0.5 * arrDict["bodybackcenter3_-1"] + 0.5 * arrDict["bodybackcenter3_1"]
    )
    arrDictNew["bodybackcenter4_0"] = (
        0.5 * arrDict["bodybackcenter4_-1"] + 0.5 * arrDict["bodybackcenter4_1"]
    )
    arrDictNew["bodybackcenter5_0"] = (
        0.5 * arrDict["bodybackcenter5_-1"] + 0.5 * arrDict["bodybackcenter5_1"]
    )

    arrDictNew["bodybackcenter1_1"] = (
        0.5 * arrDictNew["bodybackcenter1_0"] + 0.5 * arrDict["bodybackcenter1_1"]
    )
    arrDictNew["bodybackcenter2_1"] = (
        0.5 * arrDictNew["bodybackcenter2_0"] + 0.5 * arrDict["bodybackcenter2_1"]
    )
    arrDictNew["bodybackcenter3_1"] = (
        0.5 * arrDictNew["bodybackcenter3_0"] + 0.5 * arrDict["bodybackcenter3_1"]
    )
    arrDictNew["bodybackcenter4_1"] = (
        0.5 * arrDictNew["bodybackcenter4_0"] + 0.5 * arrDict["bodybackcenter4_1"]
    )
    arrDictNew["bodybackcenter5_1"] = (
        0.5 * arrDictNew["bodybackcenter5_0"] + 0.5 * arrDict["bodybackcenter5_1"]
    )

    arrDictNew["bodybackcenter1_-1"] = (
        0.5 * arrDictNew["bodybackcenter1_0"] + 0.5 * arrDict["bodybackcenter1_-1"]
    )
    arrDictNew["bodybackcenter2_-1"] = (
        0.5 * arrDictNew["bodybackcenter2_0"] + 0.5 * arrDict["bodybackcenter2_-1"]
    )
    arrDictNew["bodybackcenter3_-1"] = (
        0.5 * arrDictNew["bodybackcenter3_0"] + 0.5 * arrDict["bodybackcenter3_-1"]
    )
    arrDictNew["bodybackcenter4_-1"] = (
        0.5 * arrDictNew["bodybackcenter4_0"] + 0.5 * arrDict["bodybackcenter4_-1"]
    )
    arrDictNew["bodybackcenter5_-1"] = (
        0.5 * arrDictNew["bodybackcenter5_0"] + 0.5 * arrDict["bodybackcenter5_-1"]
    )

    # 左体侧
    if "armleftunder1" in arrDict and "armleftunder2" in arrDict:
        arrDictNew["bodyleft"] = (
            0.5 * arrDict["armleftunder1"] + 0.5 * arrDict["armleftunder2"]
        )
    elif "armleftunder1" in arrDict:
        arrDictNew["bodyleft"] = arrDict["armleftunder1"]

    # 右体侧
    if "armrightunder1" in arrDict and "armrightunder2" in arrDict:
        arrDictNew["bodyright"] = (
            0.5 * arrDict["armrightunder1"] + 0.5 * arrDict["armrightunder2"]
        )
    elif "armrightunder1" in arrDict:
        arrDictNew["bodyright"] = arrDict["armrightunder1"]

    # 左肩
    if "shoulderleft1" in arrDict:
        arrDictNew["shoulderleft"] = arrDict["shoulderleft1"]

    # 右肩
    if "shoulderright1" in arrDict:
        arrDictNew["shoulderright"] = arrDict["shoulderright1"]

    # 左袖子
    arrDictNew["armleft1"] = (
        0.5 * arrDict["armleftleft1"] + 0.5 * arrDict["armleftright1"]
    )
    arrDictNew["armleft2"] = (
        0.5 * arrDict["armleftleft2"] + 0.5 * arrDict["armleftright2"]
    )
    arrDictNew["armleft3"] = (
        0.5 * arrDict["armleftleft3"] + 0.5 * arrDict["armleftright3"]
    )

    # 右袖子
    if "armrightleft1" in arrDict and "armrightright1" in arrDict:
        arrDictNew["armright1"] = (
            0.5 * arrDict["armrightleft1"] + 0.5 * arrDict["armrightright1"]
        )
    if "armrightleft2" in arrDict and "armrightright2" in arrDict:
        arrDictNew["armright2"] = (
            0.5 * arrDict["armrightleft2"] + 0.5 * arrDict["armrightright2"]
        )
    if "armrightleft3" in arrDict and "armrightright3" in arrDict:
        arrDictNew["armright3"] = (
            0.5 * arrDict["armrightleft3"] + 0.5 * arrDict["armrightright3"]
        )

    # 左袖口（手腕）
    if "wristleft2" in arrDict and "wristleft4" in arrDict:
        arrDictNew["wristleft"] = (
            0.5 * arrDict["wristleft2"] + 0.5 * arrDict["wristleft4"]
        )

    # 右袖口（手腕）
    if "wristright2" in arrDict and "wristright4" in arrDict:
        arrDictNew["wristright"] = (
            0.5 * arrDict["wristright2"] + 0.5 * arrDict["wristright4"]
        )

    # 裙前片（下半身前）(1\-1)
    arrDictNew["pelvisfront1_-1"] = 0.75 * arrDict["skirt1"] + 0.25 * arrDict["skirt6"]
    if "leftlegcenter1_1" in arrDict:
        arrDictNew["pelvisfront2_-1"] = arrDict["leftlegcenter1_1"]
        arrDictNew["pelvisfront3_-1"] = arrDict["leftlegcenter2_1"]
        arrDictNew["pelvisfront4_-1"] = arrDict["leftlegcenter3_1"]
        arrDictNew["pelvisfront5_-1"] = arrDict["leftlegcenter4_1"]

    arrDictNew["pelvisfront1_1"] = 0.75 * arrDict["skirt1"] + 0.25 * arrDict["skirt5"]
    if "rightlegcenter1_1" in arrDict:
        arrDictNew["pelvisfront2_1"] = arrDict["rightlegcenter1_1"]
        arrDictNew["pelvisfront3_1"] = arrDict["rightlegcenter2_1"]
        arrDictNew["pelvisfront4_1"] = arrDict["rightlegcenter3_1"]
        arrDictNew["pelvisfront5_1"] = arrDict["rightlegcenter4_1"]

    # 裙后片（下半身后）(1/-1)
    arrDictNew["pelvisback1_-1"] = 0.25 * arrDict["skirt1"] + 0.75 * arrDict["skirt6"]
    if "leftlegcenter1_3" in arrDict:
        arrDictNew["pelvisback2_-1"] = arrDict["leftlegcenter1_3"]
        arrDictNew["pelvisback3_-1"] = arrDict["leftlegcenter2_3"]
        arrDictNew["pelvisback4_-1"] = arrDict["leftlegcenter3_3"]
        arrDictNew["pelvisback5_-1"] = arrDict["leftlegcenter4_3"]

    arrDictNew["pelvisback1_1"] = 0.25 * arrDict["skirt3"] + 0.75 * arrDict["skirt5"]
    if "rightlegcenter1_3" in arrDict:
        arrDictNew["pelvisback2_1"] = arrDict["rightlegcenter1_3"]
        arrDictNew["pelvisback3_1"] = arrDict["rightlegcenter2_3"]
        arrDictNew["pelvisback4_1"] = arrDict["rightlegcenter3_3"]
        arrDictNew["pelvisback5_1"] = arrDict["rightlegcenter4_3"]

    # 裙左侧片
    arrDictNew["pelvisleft1"] = arrDict["wristleft3"]
    if "leftlegcenter1_0" in arrDict:
        arrDictNew["pelvisleft2"] = arrDict["leftlegcenter1_0"]
        arrDictNew["pelvisleft3"] = arrDict["leftlegcenter2_0"]
        arrDictNew["pelvisleft4"] = arrDict["leftlegcenter3_0"]
        arrDictNew["pelvisleft5"] = arrDict["leftlegcenter4_0"]

    # 裙右侧片
    arrDictNew["pelvisright1"] = arrDict["wristright3"]
    if "rightlegcenter1_2" in arrDict:
        arrDictNew["pelvisright2"] = arrDict["rightlegcenter1_2"]
        arrDictNew["pelvisright3"] = arrDict["rightlegcenter2_2"]
        arrDictNew["pelvisright4"] = arrDict["rightlegcenter3_2"]
        arrDictNew["pelvisright5"] = arrDict["rightlegcenter4_2"]

    return arrDictNew


# 输入：板片信息；输出：大类名称和匹配到的安排点名称
def get_panel_name(ref_point_conf, pattern_bbox, ref_points, k=1.0, up_axis="Y"):
    p_label = max(ref_point_conf, key=ref_point_conf.get)

    up_axis_idx = 1 if up_axis.lower() == "y" else 2  # default is Y axis

    # check upper body or lower body
    bust_top_line = (
        0.5 * ref_points["bodyfrontcenter2_0"][up_axis_idx]
        + 0.5 * ref_points["bodyfrontcenter3_0"][up_axis_idx]
    )
    bust_bottom_line = (
        0.9 * ref_points["bodyfrontcenter3_0"][up_axis_idx]
        + 0.1 * ref_points["bodyfrontcenter4_0"][up_axis_idx]
    )
    waist_line = (
        ref_points["bodyfrontcenter4_0"][up_axis_idx] * 0.75
        + ref_points["bodyfrontcenter3_0"][up_axis_idx] * 0.25
    )
    hip_line = ref_points["pelvisfront1_1"][1]
    ankle_line = (
        ref_points["pelvisleft5"][1] * 0.5 + ref_points["pelvisright5"][1] * 0.5
    )

    pattern_scale = pattern_bbox[1] - pattern_bbox[0]
    pattern_flatness = pattern_scale[up_axis_idx] ** 3 / np.prod(pattern_scale)

    _top = pattern_bbox[1][up_axis_idx]  # cloth piece top
    _bottom = pattern_bbox[0][up_axis_idx]  # cloth piece bottom

    if p_label.startswith("body") or p_label.startswith("pelvis"):
        if _top < bust_bottom_line and _bottom < hip_line:
            p_label = p_label.replace("body", "pelvis")
        if _top > bust_top_line:
            p_label = p_label.replace("pelvis", "body")

    if (
        p_label.startswith("body") or p_label.startswith("pelvis")
    ) and pattern_flatness < k:
        if p_label.endswith("left") or p_label.endswith("right"):
            p_label = p_label.replace("body", "wrist").replace("pelvis", "wrist")
        if _bottom > hip_line and _top < waist_line:
            p_label = "waist"
        elif _bottom < ankle_line:
            p_label = "ankle"
        elif _bottom > bust_top_line:
            p_label = "neck"

    return p_label


# 输入：第一层级分类和安排点名称；输出：第二层级分类（最终保存到json里的label）
def get_panel_label_by_name(PClass, PName):
    p_label = ""
    if (
        PClass != "bodyfront" and PClass != "bodyback"
    ):  # 直接继承第一层级的（语雀里有写）
        p_label = PClass
    else:
        if PName.endswith("_-1"):  # -1表示左
            p_label = PClass + "left"
        elif PName.endswith("_0"):  # 0表示中
            p_label = PClass + "center"
        elif PName.endswith("1"):  # 1表示右
            p_label = PClass + "right"

    # 输出之前检查一下是否为空字符串
    if p_label == "":
        # print(PClass,"  "," 的PLabel为空！")
        p_label == "head"
    return p_label


def auto_labelling(obj_fp, output_fp, ref_id="arrangements", _CMAP=None, glctx=None):
    try:
        _tic = time.perf_counter()

        print(">> ", obj_fp)
        # tic = time.perf_counter()
        data_item_dir = os.path.dirname(obj_fp)
        result_fp = os.path.join(data_item_dir, "panels.json")
        # if os.path.exists(result_fp) and os.path.getsize(result_fp) > 0:
        #     print('[DONE] Skipping %s'%(obj_fp))
        #     return True

        # categories = list(_CMAP.keys())

        avatar_fp = os.path.join(data_item_dir, "avatar.json")
        smd_fp = os.path.join(data_item_dir, "smd.json")

        assert os.path.exists(obj_fp), "Cannot find .obj in %s" % (data_item_dir)
        assert os.path.exists(avatar_fp), "Cannot find avatar.json in %s" % (
            data_item_dir
        )
        assert os.path.exists(smd_fp), "Cannot find smd.json in %s" % (data_item_dir)

        with open(avatar_fp, "rb") as f:
            avatar_data = json.load(f)
            arrDict = {}
            for item in avatar_data[0][ref_id]:
                arrDict[item["name"].lower()] = np.array(item["xyz"], dtype=np.float32)

            # print(arrDict["HeadV3"])
            # 如果arrangepoint是ArrangePoint+数字开头，要先进行转换成标准的安排点
            # arrList = np.array(list(arrDict.values()))
            arrDictNew = get_arrange_pts(arrDict)
            arrList = np.array(list(arrDictNew.values()))
            arrKeys = list(arrDictNew.keys())

        with open(smd_fp, "rb") as f:
            smd_data = json.load(f)
        cloth_patterns, canvas_bbox = json2img(
            smd_data, os.path.join(data_item_dir, "smd.png"), img_dpi=300, vis=True
        )
        # print("*** cloth patterns: \n", cloth_patterns)

        # 读取obj文件
        mesh_obj = read_obj(obj_fp)
        # print('Load mesh: ', np.min(mesh_obj.points, axis=0), np.max(mesh_obj.points, axis=0))

        panelList = []  # 创建一个panelsList用于保存一个个字典

        global_centroid = np.mean(mesh_obj.points, axis=0)
        global_scale = (
            np.max(mesh_obj.points, axis=0)[1] - np.min(mesh_obj.points, axis=0)[1]
        )

        for cell_idx, cell in enumerate(mesh_obj.cells):  # 遍历所有板片
            uuid = np.array(mesh_obj.field_data["obj:group_tags"])[
                cell_idx
            ]  # 先开辟一个字段保存uuid
            if uuid not in cloth_patterns:
                continue

            # print('*** uuid: ', uuid)

            valid_verts = mesh_obj.points[np.unique(cell.data), :]
            centroid = np.mean(valid_verts, axis=0)

            bbox = [np.min(valid_verts, axis=0), np.max(valid_verts, axis=0)]

            bias = (centroid - global_centroid) / global_scale

            arrDistances = np.linalg.norm(arrList - centroid, axis=1)

            dist_conf = 1.0 / arrDistances
            dist_conf = dist_conf / np.sum(dist_conf)

            ref_point_idx = arrDistances.argsort()[:4]  # 3 nearest points

            ref_point_conf = {}
            for p_idx in ref_point_idx:
                ref_point_id = get_pattern_cls(arrKeys[p_idx])
                if ref_point_id not in ref_point_conf:
                    ref_point_conf[ref_point_id] = 0.0
                ref_point_conf[ref_point_id] += dist_conf[p_idx]

            # print('*** ref_point_conf:', uuid, ref_point_conf)

            # 创建一个json对象（字典），表示单个板片信息
            singlePanel = {}

            singlePanel["uuid"] = uuid
            singlePanel["points"] = cloth_patterns[uuid].tolist()

            p_label = get_panel_name(ref_point_conf, bbox, arrDictNew)
            p_label = ru.mapping_pattern_label(p_label)

            if np.all(bias < 0.75):
                singlePanel["label"] = p_label
            else:
                singlePanel["label"] = "none"

            # print("*** p_label: ", p_label)

            singlePanel["color"] = _CMAP[p_label].tolist()
            singlePanel["xSpouse"] = None  # 左右配偶先置空，让标注师指定
            singlePanel["ySpouse"] = None  # 前后配偶先置空，让标注师指定
            panelList.append(
                singlePanel
            )  # 全部字段都开辟之后把这个字典加入之前定义的List

        if glctx is not None:
            ru.render_anno_dr(
                mesh_obj,
                dict(
                    [
                        (panel_item["uuid"], panel_item["label"])
                        for panel_item in panelList
                    ]
                ),
                _CMAP,
                output_fp,
                cam_radius=2.0,
                mesh_scale=0.001,
                reso=800,
                glctx=glctx,
            )
        else:
            ru.render_anno_plt(
                mesh_obj,
                dict(
                    [
                        (panel_item["uuid"], panel_item["label"])
                        for panel_item in panelList
                    ]
                ),
                _CMAP,
                output_fp,
            )

        # 最后将之前那个List保存成json，看了文档发现外面还包了一层panels，所以还要创建一个键值对
        panels = {}
        panels["panels"] = panelList
        panels["bbox"] = canvas_bbox

        with open(result_fp, "w") as f:
            f.write(json.dumps(panels))

        _toc = time.perf_counter()
        print("[DONE] Processing time for %s: %.4f s." % (obj_fp, _toc - _tic))
        return True

    except Exception as e:
        print("[FAILED] %s" % (obj_fp), e)
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processing *.sproj -> cloth piece obj"
    )
    parser.add_argument(
        "-n", "--name", default="Auto Labelling", type=str, help="Input directory."
    )
    parser.add_argument(
        "-d", "--dir", default=".\dresses", type=str, help="Input directory."
    )
    parser.add_argument(
        "-r",
        "--ref",
        default="arrangements",
        type=str,
        help="Reference points, choose between arrangement points and body joints.",
    )
    parser.add_argument(
        "-o", "--output", default=None, type=str, help="Output directory."
    )
    parser.add_argument(
        "-p",
        "--pool",
        default=32,
        type=int,
        help="Number of threads for multi processing.",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        default=1000,
        type=int,
        help="Number of threads for multi processing.",
    )
    parser.add_argument(
        "--use_dr", action="store_true", help="Whether to use nvdiffrast renderer."
    )
    parser.add_argument(
        "--config", default="info.json", type=str, help="Path to info.json."
    )
    parser.add_argument(
        "--ignore_exist", action="store_true", help="Whether to ignore processed items."
    )
    args, cfg_cmd = parser.parse_known_args()

    data_root = args.dir
    # if args.output is None: args.output = os.path.join(data_root, '..', 'vis')
    # os.makedirs(args.output, exist_ok=True)

    all_items = sorted(glob(os.path.join(data_root, "**", "*.obj"), recursive=True))

    if args.ignore_exist:
        all_items = [
            x
            for x in all_items
            if not os.path.exists(os.path.join(os.path.dirname(x), "panels.json"))
        ]
    print("Total number of items: ", len(all_items))

    ref_id = args.ref  # 这里是arrangement

    profile_file = json.load(open(args.config, "rb"))
    _CMAP = profile_file["pattern_cls"]
    print(_CMAP)
    for cls_key in _CMAP:
        print(cls_key, _CMAP[cls_key])
        _CMAP[cls_key] = np.array(matplotlib.colors.to_rgba(_CMAP[cls_key]["color"]))
        # _CMAP[cls_key][-1] = 0.8
        print(_CMAP[cls_key])

    print(
        "Parsing labels, find %d classes: \n\t%s"
        % (len(_CMAP), "\n\t".join(_CMAP.keys()))
    )

    date_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    init_msg = f"### [{date_str}] {args.name}\n---\n"
    init_msg += f"- **Input Dir  :** {args.dir}\n"
    init_msg += f"- **Total Items:** {len(all_items)}\n"
    init_msg += f"- **Output Dir :** {args.output}\n"
    init_msg += f"- **Batch Size :** {args.batchsize}\n"
    init_msg += f"- **Num Pools  :** {args.pool}\n"

    init_msg = markdown_msg("Init Run", init_msg)
    ding_msg(init_msg)

    if args.use_dr:
        tic = time.perf_counter()
        glctx = dr.RasterizeGLContext()
        print("Init nvdiffrast renderer: %.4f s" % (time.perf_counter() - tic))

        succeed_cnt = 0
        for idx, input_file in enumerate(all_items):
            output_file = (
                os.path.join(os.path.dirname(input_file), "labelling_vis.png")
                if args.output is None
                else os.path.join(
                    args.output, os.path.basename(input_file).replace(".obj", ".png")
                )
            )
            succeed_cnt += int(
                auto_labelling(input_file, output_file, ref_id, _CMAP, glctx=glctx)
            )

            if idx % args.batchsize == 0:
                msg = text_msg(
                    "[%03d / %03d] info: succeed %d; failed %d. \n[%s] input: %s"
                    % (
                        idx + 1,
                        len(all_items),
                        succeed_cnt,
                        idx + 1 - succeed_cnt,
                        args.name,
                        args.dir,
                    )
                )
                ding_msg(msg)

    else:
        params = [
            (
                x,
                os.path.join(os.path.dirname(x), "labelling_vis.png")
                if args.output is None
                else os.path.join(
                    args.output, os.path.basename(x).replace(".obj", ".png")
                ),
                ref_id,
                _CMAP,
            )
            for x in all_items
        ]
        batch_params = [
            params[i : i + args.batchsize]
            for i in range(0, len(params), args.batchsize)
        ]

        succeed_cnt = 0
        for batch_idx, batch in enumerate(batch_params):
            with Pool(args.pool) as p:
                results = p.starmap(auto_labelling, batch)

            batch_succeed_cnt = sum(results)
            succeed_cnt += batch_succeed_cnt
            msg = text_msg(
                "[%03d / %03d] batch info: total %d; succeed %d; failed %d. \n[%s] input: %s"
                % (
                    batch_idx,
                    len(batch_params),
                    len(batch),
                    batch_succeed_cnt,
                    len(batch) - batch_succeed_cnt,
                    args.name,
                    args.dir,
                )
            )
            ding_msg(msg)

            if args.output is None:
                args.output = args.dir
            with open(os.path.join(args.output, "app.log"), "a") as f:
                f.writelines(
                    ["%s\t %d\n" % (x[0], y) for x, y in zip(batch, results) if y]
                )

    date_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    finilize_msg = f"### [{date_str}] {args.name} finished.\n---\n"
    finilize_msg += f"- **Input Dir  :** {args.dir}\n"
    finilize_msg += f"- **Output Dir :** {args.output}\n"
    finilize_msg += f"- **Total Items:** {len(all_items)}\n"
    finilize_msg += f"- **Valid Items:** {succeed_cnt}\n"
    finilize_msg = markdown_msg("Finished.", finilize_msg)
    ding_msg(finilize_msg)
