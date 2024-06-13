import json
from shutil import copyfile
from typing import List
from zipfile import ZipFile
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import numpy as np

from glob import glob

import multiprocessing as mulpc

SeqEdgeType = {
    "OutLine": 0,  # DXF导入缝边 layerId=1
    "GrainLine": 1,  # 面料方向线 layerId=7
    "BaseLine": 2,  # 基本线 layerId=8
    "SewLine": 3,  # 净边 layerId=14
    "InnerLine": 4,  # 内部可编辑边 layerId=8 通过导入选项和基本线区分
    "MirrorLine": 5,  # 镜像线 layerId=6
    "SeamAllowance": 6,  # 系统缝份 layerId=1
    "ReferenceLine": 7,  #
    "BindingLine": 8,  # 贴边线
    "Invalid": 999999,
}

COLOR_TABLE = {}

SeqEdgeTypeNeed = ["SewLine"]


def poly_area(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def smd2zip(smd_path, save_dir=None):
    assert smd_path.endswith(".smd")
    smd_name = os.path.basename(smd_path)
    if save_dir is None:
        smd_dir = os.path.dirname(smd_path)
        save_dir = os.path.join(smd_dir, "output")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    new_file_name = os.path.splitext(smd_name)[0] + ".zip"  # 生成新的文件名
    new_file_path = os.path.join(save_dir, new_file_name)
    copyfile(smd_path, new_file_path)  # 重命名文件
    return new_file_path


def zip2img(zip_path, save_path=None, **kwargs):
    assert zip_path.endswith(".zip")
    js = read_SmdJsonInZip(zip_path)
    json2img(js, save_path, **kwargs)


def read_SmdJsonInZip(path):
    zp = ZipFile(path)
    js_str = zp.read("smd.json").decode("utf-8")
    return json.loads(js_str)


def drawLines(points, color="black"):
    # plt.scatter(points[:, 0], points[:, 1], color=color, s=1)
    plt.plot(points[:, 0], points[:, 1], color=color, linewidth=1.0)


def transform2D(points: List = None, trans_matrix: List = None):
    assert points or trans_matrix
    points = np.array(points, dtype=np.float32)
    trans_matrix = np.array(trans_matrix).reshape((4, 4))
    points = np.concatenate([points, np.ones_like(points[:, :1])], axis=-1)
    result = np.matmul(points, trans_matrix)
    return result / result[:, 3:]


def json2img(
    js,
    save_path=None,
    img_dpi=100,
    vis=True,
    padding=10,
    draw_poly=False,
    edgecolors=None,
    facecolors=None,
):
    seqEdgeTypeNeedId = [SeqEdgeType[v] for v in SeqEdgeTypeNeed]

    clothPieces = js["garment"]["clothPieces"]
    patterns = js["garment"]["patterns"]
    connectPairs = js["garment"]["connectPairs"]

    if edgecolors is None:
        colordict = {}
        for connectPair in connectPairs:
            color = tuple(connectPair["color"])
            seamEdgeGstartA = connectPair["seamEdgeGroupA"]["instancedSeamEdges"][0][
                "seamEdge"
            ]["seqSegment"]["start"]["edgeId"]
            seamEdgeGstartB = connectPair["seamEdgeGroupB"]["instancedSeamEdges"][0][
                "seamEdge"
            ]["seqSegment"]["start"]["edgeId"]
            colordict.update({seamEdgeGstartA: color, seamEdgeGstartB: color})
    else:
        colordict = edgecolors

    symmetryPairs = js["garment"]["symmetryPairs"]
    symmetryEdgeId = [
        i["symmetryEdgeId"] for i in symmetryPairs if i["symmetryType"] == 2
    ]

    cloth_edge_points = {}

    x_min, y_min = 999999, 999999
    x_max, y_max = -999999, -999999

    for cloth in clothPieces:
        uuid = cloth["id"]
        patternId = cloth["patternId"]
        transform2D_matrix = cloth["transform2D"]
        edgesList = [
            j["edges"]
            for i in patterns
            for j in i["seqEdges"]
            if i["id"] == patternId and j["type"] in seqEdgeTypeNeedId
        ]

        cloth_edge_points[uuid] = []
        for edges in edgesList:
            for edge in edges:
                if edge["id"] in symmetryEdgeId:
                    continue
                controlPoints = edge["controlPoints"]
                transPoints = transform2D(controlPoints, transform2D_matrix)[:, :2]
                cloth_edge_points[uuid].append(transPoints)

                _x_min, _y_min = transPoints.min(axis=0)
                _x_max, _y_max = transPoints.max(axis=0)

                x_min, y_min = min(x_min, _x_min), min(y_min, _y_min)
                x_max, y_max = max(x_max, _x_max), max(y_max, _y_max)

                if vis:
                    drawLines(transPoints, color=colordict.get(edge["id"]))

        cloth_edge_points[uuid] = np.concatenate(cloth_edge_points[uuid], axis=0)

    x_min, y_min = x_min - padding, y_min - padding
    x_max, y_max = x_max + padding, y_max + padding

    for uuid in cloth_edge_points:
        cloth_edge_points[uuid] = (
            cloth_edge_points[uuid] - np.array([[x_min, y_min]])
        ) / (np.array([[x_max, y_max]]) - np.array([[x_min, y_min]]) + 1e-8)

    # Merge symmetrical cloth pieces
    symDict = dict(
        [
            (x["clothPieceBId"], x["clothPieceAId"])
            for x in symmetryPairs
            if x["symmetryType"] == 2
        ]
    )
    for uuid in symDict.keys():
        if uuid in symDict:
            if uuid in cloth_edge_points and symDict[uuid] in cloth_edge_points:
                cloth_edge_points[symDict[uuid]] = np.concatenate(
                    [cloth_edge_points[symDict[uuid]], cloth_edge_points[uuid][::-1]],
                    axis=0,
                )
                del cloth_edge_points[uuid]

    if vis:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        plt.axis("off")

        if save_path is not None:
            plt.savefig(
                save_path,
                bbox_inches="tight",
                pad_inches=0,
                transparent=True,
                dpi=img_dpi,
            )
        else:
            plt.show()
        plt.clf()

        if draw_poly:
            _, ax = plt.subplots()
            ax.axis("off")
            for p_idx, panel in enumerate(cloth_edge_points.keys()):
                poly_color = (
                    facecolors[panel] if facecolors is not None else np.random.rand(3)
                )
                ax.add_patch(
                    Polygon(
                        cloth_edge_points[panel],
                        closed=True,
                        facecolor=poly_color,
                        alpha=1.0,
                    )
                )
                if (
                    poly_area(
                        cloth_edge_points[panel][:, 0], cloth_edge_points[panel][:, 1]
                    )
                    > 1e-3
                ):
                    center_point = np.mean(cloth_edge_points[panel], axis=0)
                    ax.text(
                        center_point[0],
                        center_point[1],
                        "%d" % (p_idx + 1),
                        fontsize=10,
                        horizontalalignment="center",
                        verticalalignment="center_baseline",
                        bbox=dict(facecolor="white", alpha=0.25, edgecolor="none"),
                    )

            if save_path is not None:
                plt.savefig(
                    save_path.replace(".png", "_poly.png"),
                    bbox_inches="tight",
                    pad_inches=0,
                    transparent=True,
                    dpi=img_dpi,
                )
            else:
                plt.show()

        plt.clf()

    return cloth_edge_points, (x_min, y_min, x_max, y_max)


def smd2img_single(
    file_path, save_dir=None, is_showImg=True, is_saveImg=False, **kwargs
):
    dir_path = os.path.dirname(file_path)
    smd_name = os.path.basename(file_path)
    prefix_name = smd_name.split(".")[0]
    new_smd_name = prefix_name + ".png"
    if save_dir is None:
        save_dir = dir_path
    zip_save_dir = os.path.join(save_dir, "output")
    img_save_dir = os.path.join(save_dir, "output", "img")
    os.makedirs(zip_save_dir, exist_ok=True)
    os.makedirs(img_save_dir, exist_ok=True)
    img_save_path = os.path.join(img_save_dir, new_smd_name)
    new_file_path = smd2zip(file_path, zip_save_dir)
    zip2img(
        new_file_path,
        img_save_path,
        is_saveImg=is_saveImg,
        is_showImg=is_showImg,
        **kwargs,
    )


def smd2img_thread(smd_fp):
    try:
        with open(smd_fp, "rb") as f:
            smd_data = json.load(f)
        json2img(smd_data, smd_fp.replace(".json", ".png"), 300, True)
        print(f"[DONE] {smd_fp.replace('.json', '.png')}")
        return True

    except Exception as e:
        print(f"{smd_fp} error: {e}")
        return False


def deleteZip(dir_path):
    for root, dirs, files in os.walk(dir_path):
        if not files:
            continue
        for path in files:
            if "output" in path:
                break
            if path.endswith(".zip"):
                zip_path = os.path.join(root, path)
                os.remove(zip_path)


if __name__ == "__main__":
    smd_dir = "E:\lry\code\garment_pattern_lib\data"
    smd_files = glob(os.path.join(smd_dir, "**", "smd.json"), recursive=True)
    print(len(smd_files))

    with mulpc.Pool(1) as pool:
        batch_res = pool.map(smd2img_thread, smd_files)
    print(batch_res)
    exit()
