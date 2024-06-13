import json
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import shutil
import yaml
from matplotlib.patches import Polygon, Rectangle

from constants import *
from thirdparty.garment_helper.render_helper import mapping_pattern_label


def get_export_jsons(folder: str) -> list[str]:
    # find all json paths of export result in folder,
    # return a list of json path
    files = glob.glob(f"{folder}/*.json")
    return files


def summary_anno(folder):
    jsons = get_export_jsons(folder)

    # new_jsons = []

    # for j in jsons:
    #     name_id = int(os.path.basename(j).split('.')[0].split('_')[1])
    #     if name_id < 264:
    #         new_jsons.append(j)
    
    # jsons = new_jsons

    choices = {"0": 0, "1": 0, "2": 0, "3": 0, "None": 0}

    total_task = 0

    sum_dict = {}

    # check every project
    for j in jsons:
        with open(j, "r") as f:
            j_content = json.load(f)

            # check every cloth task
            for task_content in j_content:
                task_id = task_content["meta"]["id"]
                # sum_dict[task_id] = {"0": 0, "1": 0, "2": 0, "3": 0, "None": 0}

                total_task += 1
                anno_results = task_content["annotations"][0]["result"]

                choice_flag = 0

                # check every panel of specific cloth
                for anno in anno_results:
                    if anno["type"] == "choices":
                        choice = anno["value"]["choices"][0]
                        choices[choice] += 1
                        choice_flag = 1
                        # sum_dict[task_id][choice] += 1
                        sum_dict[task_id] = choice

                if choice_flag != 1:
                    choices["None"] += 1

    print(
        f"""
          Total task: {total_task},
          Choice Summary:
            0: {choices['0']},
            1: {choices['1']},
            2: {choices['2']},
            3: {choices['3']}
          No Choice: {choices['None']}
        """
    )

    with open("/home/gyy/code/garment_pattern_lib/data/summary.json", "w") as f:
        json.dump(sum_dict, f)


def convert_to_yolo_format(bbox: list) -> list:
    # xmin, ymin, xmax, ymax -> x_center, y_center, width, height
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    x_center = xmin + width / 2
    y_center = ymin + height / 2
    return np.round([x_center, y_center, width, height], 6).tolist()


def draw_panel(
    panel_json: str,
    out_img_pth: str = None,
    out_label_pth: str = None,
    img_dpi=300,
    draw_bbox=False,
    draw_num=False,
    save_image=True,
    save_label_txt=True
):
    def poly_area(x, y): return 0.5 * np.abs(
        np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
    )

    with open(panel_json, "r") as f:
        panel_data = json.load(f)

    cloth_edge_points = dict(
        [(x["uuid"], [x["label"], np.array(x["points"])])
         for x in panel_data["panels"]]
    )

    _, ax = plt.subplots()
    ax.axis("off")

    bboxes = []

    for p_idx, panel in enumerate(cloth_edge_points.keys()):
        p_area = poly_area(
            cloth_edge_points[panel][1][:, 0], cloth_edge_points[panel][1][:, 1]
        )
        if cloth_edge_points[panel][0] == "none":   # ignore label:none
            continue
        if p_area <= 1e-3:                          # ignore small piece
            continue

        polygon = Polygon(
            cloth_edge_points[panel][1], closed=True, facecolor=POLY_COLOR
        )
        ax.add_patch(polygon)

        cls = ANNO_CLS_T[mapping_pattern_label(cloth_edge_points[panel][0])]

        points = cloth_edge_points[panel][1]
        xmin, ymin = np.min(points, axis=0)
        xmax, ymax = np.max(points, axis=0)

        bbox = [cls] + convert_to_yolo_format([xmin, 1 - ymax, xmax, 1 - ymin])
        bboxes.append(bbox)

        if draw_bbox:
            rect = Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=None, edgecolor="red"
            )
            ax.add_patch(rect)
        if draw_num and p_area > 1e-3:
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
    
    if save_image:
        plt.savefig(
            out_img_pth, bbox_inches="tight", pad_inches=0, transparent=True, dpi=img_dpi
        )

    plt.cla()
    plt.clf()
    plt.close()

    if save_label_txt and out_label_pth:
        with open(out_label_pth, "w") as f:
            for i in bboxes:
                line = " ".join(map(str, i))
                line.replace(line[-1], "")
                f.write(line + "\n")


def analyse_task(
    task_content: dict, panel_folder_list: str, output_folder: str, choices: str = "0"
):
    uuid = task_content["meta"]["id"]
    save_path = f"{output_folder}/{uuid}.json"
    if os.path.exists(save_path):
        return False

    anno_results = task_content["annotations"][0]["result"]

    flag = 0
    for anno in anno_results:
        if anno["type"] == "choices":
            flag = 1
            if anno["value"]["choices"][0] != choices:
                # print(f'Choice != 0')
                return False
    if flag == 0:
        # print(f'No choice')
        return False

    for panel_folder in panel_folder_list:
        old_panel_pth = os.path.join(panel_folder, uuid, "panels.json")
        if os.path.exists(old_panel_pth):
            break

    try:
        with open(old_panel_pth, "r") as f:
            panel_content = json.load(f)
    except Exception as e:
        return False

    origin_panels = panel_content["panels"]
    origin_bbox = panel_content["bbox"]

    anno_dict = {}
    for anno in anno_results:
        if anno["type"] == "polygonlabels":
            panel_id = anno["id"]
            label = anno["value"]["polygonlabels"][0]
            anno_dict[panel_id] = label

    new_panels = []
    for panel in origin_panels:
        if panel["uuid"] in anno_dict:
            panel["label"] = anno_dict[panel["uuid"]]
            new_panels.append(panel)

    final_panel_res = {"panels": new_panels, "bbox": origin_bbox}

    # print(f'Save to {save_path}.')
    with open(save_path, "w") as f:
        json.dump(final_panel_res, f)
    return True


def panels_to_yolo(panel_folder: str):
    yolo_folder = os.path.join(panel_folder, "yolo")
    image_folder = os.path.join(yolo_folder, "images")
    label_folder = os.path.join(yolo_folder, "labels")
    os.makedirs(yolo_folder, exist_ok=True)
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    panels = glob.glob(f"{panel_folder}/*.json")
    for panel_pth in tqdm(panels):
        uuid = os.path.basename(panel_pth).split(".")[0]
        out_img_path = os.path.join(image_folder, f"{uuid}.jpg")
        out_label_path = os.path.join(label_folder, f"{uuid}.txt")
        draw_panel(panel_pth, out_img_path, out_label_path, img_dpi=300)


def task_formatter(panel_folder_list, export_folder, output_folder, choices="0"):
    # project[task] -> panels.json
    export_jsons = get_export_jsons(export_folder)

    for ejson in tqdm(export_jsons):
        # 读json
        with open(ejson, "r") as f:
            json_content = json.load(f)
        # 分割每个task
        for task in json_content:
            # 摘取task json需要的信息
            analyse_task(task, panel_folder_list, output_folder, choices)


def yolo_spliter(root_folder, out_folder, split_factor=0.8):
    images_folder = os.path.join(root_folder, "images")
    labels_folder = os.path.join(root_folder, "labels")

    imgs_list = list(sorted(os.listdir(images_folder)))
    idxs = list(range(len(imgs_list)))
    np.random.shuffle(idxs)

    train_idx = idxs[: int(split_factor * len(idxs))]
    val_idx = idxs[int(split_factor * len(idxs)):]

    img_train = os.path.join(out_folder, "images", "train")
    img_val = os.path.join(out_folder, "images", "val")
    label_train = os.path.join(out_folder, "labels", "train")
    label_val = os.path.join(out_folder, "labels", "val")

    os.makedirs(img_train, exist_ok=True)
    os.makedirs(img_val, exist_ok=True)
    os.makedirs(label_train, exist_ok=True)
    os.makedirs(label_val, exist_ok=True)

    for i in train_idx:
        shutil.copy(
            os.path.join(images_folder, imgs_list[i]),
            os.path.join(img_train, imgs_list[i]),
        )
        shutil.copy(
            os.path.join(labels_folder, imgs_list[i].replace("jpg", "txt")),
            os.path.join(label_train, imgs_list[i]).replace("jpg", "txt"),
        )

    for j in val_idx:
        shutil.copy(
            os.path.join(images_folder, imgs_list[j]),
            os.path.join(img_val, imgs_list[j]),
        )
        shutil.copy(
            os.path.join(labels_folder, imgs_list[j].replace("jpg", "txt")),
            os.path.join(label_val, imgs_list[j]).replace("jpg", "txt"),
        )

    yolo_format = dict(path=out_folder, train=img_train,
                       val=img_val, names=ANNO_CLS)

    with open(os.path.join(out_folder, "yolo.yaml"), "w") as outfile:
        yaml.dump(yolo_format, outfile, default_flow_style=False)


def main_task_reformater(export_folder, cloth_types: list[str], choices="0") -> str:
    # input:  [project json] (exported from label studio)
    # output: [panels.json]
    panel_folder_list = [
        f"/data/AIGP/style3D_data/{i}/objs" for i in cloth_types]

    output_folder = export_folder.replace(
        "label_studio_anno_export", "annotated")
    output_folder = f"{output_folder}/choices_{choices}"
    os.makedirs(output_folder, exist_ok=True)

    task_formatter(panel_folder_list, export_folder,
                   output_folder, choices=choices)
    # summary_anno(export_folder)

    return output_folder


if __name__ == "__main__":
    export_folder = "/home/gyy/code/garment_pattern_lib/data/label_studio_anno_export/exported_from_code_20231122"
    
    # # 1. 将project中的结果转换成panels.json
    # cloth_types = ["dress", "coat", "top", "pant"]
    # print("Step 1: convert to panels.json...")
    # print(f"Cloth Types: {cloth_types}.")
    # panels_folder = main_task_reformater(export_folder, cloth_types, choices="0")
    # print(panels_folder)

    # # 2. panels.json转换成(image, yolo_anno.txt)
    # print("Step2: convert panels.json to yolo...")
    # # panels_folder = f"{panels_folder}/choices_0"
    # panels_to_yolo(panels_folder)

    # # 3. 8:2分割train和val集
    # # panels_folder = "/home/gyy/code/garment_pattern_lib/data/annotated/exported_from_code/choices_0"
    # print(f'Step3: split train and val dataset...')
    # root_folder = f"{panels_folder}/yolo"
    # output_folder = f"{panels_folder}/split"
    # yolo_spliter(root_folder, output_folder)

    summary_anno(export_folder)