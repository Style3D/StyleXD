import os
import matplotlib
import numpy as np
import requests
import json
import nvdiffrast.torch as dr

from .obj_helper import read_obj
from .file_helper import (
    dataset_reader,
    studio_to_panel,
    time_getter,
    delete_old_labelling_vis,
)
from .render_helper import render_anno_dr, mapping_pattern_label

ANNO_FOLDER = "style3D_data_anno"


class HookHelper:
    def __init__(self, cmap, data, root, cfgs, data_url) -> None:
        self.glctx = dr.RasterizeGLContext()
        self.cmap = cmap
        for cls_key in self.cmap:
            self.cmap[cls_key] = np.array(
                matplotlib.colors.to_rgba(self.cmap[cls_key]["color"])
            )
        self.data = data
        self.file_urls = dataset_reader(self.data)
        self.root = root
        self.url = data_url
        self.cfgs = cfgs

    def _update_task(self, studio_json, output_list):
        task_data = studio_json["task"]

        task_data["data"]["image_f"] = output_list[0]
        task_data["data"]["image_l"] = output_list[1]
        task_data["data"]["image_b"] = output_list[2]
        task_data["data"]["image_r"] = output_list[3]

        update_task_data = {"data": task_data["data"]}

        headers = {
            "Authorization": f"Token {self.cfgs['labelstudio']['key']}",
            "Content-Type": "application/json",
        }

        response = requests.patch(
            f"{self.cfgs['labelstudio']['url']}/api/tasks/{task_data['id']}",
            json=update_task_data,
            headers=headers,
        )

        if response.status_code == 200:
            print("[INFO]UPDATE_TASK send successfully!")
            if self.new_panel_folder:
                delete_old_labelling_vis(self.new_panel_folder)
            # print(response.json())
        else:
            print("[Warning]Error creating webhook:", response.text)

    def _rerender(self) -> list:
        obj_fp = os.path.join(self.item_dir, os.path.basename(self.item_dir) + ".obj")
        panel_fp = os.path.join(self.new_panel_folder, "panels.json")
        output_fp = os.path.join(
            self.new_panel_folder, f"labelling_vis_{time_getter()}.png"
        )  # save in anno folder

        # load annotations
        with open(panel_fp, "rb") as f:
            data = json.load(f)
            annotations = dict(
                [(panel["uuid"], panel["label"]) for panel in data["panels"]]
            )

        # load obj
        mesh_obj = read_obj(obj_fp)

        output_list = render_anno_dr(
            mesh_obj,
            annotations,
            self.cmap,
            output_fp,
            glctx=self.glctx,
            anno_mapping_fn=mapping_pattern_label,
        )

        for i in range(len(output_list)):
            output_list[i] = output_list[i].replace(self.root, self.url)

        return output_list

    def file_change(self, studio_json: dict):
        # get panel json
        panel_id = studio_json["task"]["meta"]["id"]

        panel_file_pth = None
        for url in self.file_urls:
            if panel_id in url and "panels" in url:
                panel_file_pth = url
                break

        print(f"panel_file_pth: {panel_file_pth}")

        if panel_file_pth is None:
            raise FileNotFoundError

        # 1. change panels.json
        print("[INFO]Receive change and update panels.json.")
        origin_panels = json.loads(requests.get(panel_file_pth).content)
        new_panel_content = studio_to_panel(studio_json, origin_panels)

        dst_pth = panel_file_pth
        local_panel_path = dst_pth.replace(self.url, self.root)

        local_panel_folder = local_panel_path[:-12]
        self.new_panel_folder = local_panel_folder.replace(
            "style3D_data/", f"style3D_data/{ANNO_FOLDER}/"
        )
        print(f"[INFO]Save new anno in: {self.new_panel_folder}")
        os.makedirs(self.new_panel_folder, exist_ok=True)

        new_json_path = f"{self.new_panel_folder}/panels.json"
        with open(new_json_path, "w") as f:
            json.dump(new_panel_content, f)
            print(f"[INFO]Successfully save result in {new_json_path}.")

        # 2. rerender
        self.item_dir = local_panel_folder
        output_list = self._rerender()

        self._update_task(studio_json, output_list)
