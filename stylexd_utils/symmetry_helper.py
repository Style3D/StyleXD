import json
import os
from tqdm import tqdm
from thirdparty.garment_helper.smd_helper import json2img


def edit_symmetry(smd_pth, panel_pth):
    # 根据smd.json更新panels.json中的points
    with open(panel_pth, "r") as f:
        panel_data = json.load(f)

    with open(smd_pth, "r") as f:
        smd_data = json.load(f)

    cloth_edge_points, _ = json2img(smd_data, vis=False)

    p_dict = panel_data["panels"]

    for i in range(len(p_dict)):
        p_dict[i]["points"] = cloth_edge_points[p_dict[i]["uuid"]].tolist()

    return panel_data


def symmetry_helper(root_folder, reference_folder=None):
    # panels.json -> panels_old.json
    # symmetry: panels.json
    for task_folder in tqdm(os.listdir(root_folder)):
        task_folder = os.path.join(root_folder, task_folder)
        if reference_folder:
            smd_folder = os.path.join(reference_folder, task_folder)
            smd_path = os.path.join
            (smd_folder, "smd.json")


        else:
            smd_path = os.path.join(task_folder, "smd.json")
        panel_path = os.path.join(task_folder, "panels.json")

        if os.path.exists(panel_path.replace(".json", "_old.json")):
            print(f"[INFO]{task_folder} has been processed.")
            continue

        if os.path.exists(smd_path) and os.path.exists(panel_path):
            symmetry_panel_data = edit_symmetry(smd_path, panel_path)

            os.rename(panel_path, panel_path.replace(".json", "_old.json"))

            with open(panel_path, "w") as f:
                json.dump(symmetry_panel_data, f)


def exported_symmetry_helper(panel_folder, smd_root_folder):
    json_list = os.listdir(panel_folder)
    for j_pth in tqdm(json_list):
        uuid = j_pth.split('.')[0]
        panel_path = os.path.join(panel_folder, j_pth)
        smd_path = os.path.join(smd_root_folder, uuid, 'smd.json')
        
        if os.path.exists(smd_path) and os.path.exists(panel_path):
            symmetry_panel_data = edit_symmetry(smd_path, panel_path)
        
            # os.rename(panel_path, panel_path.replace('.json', '_old.json'))
            
            with open(panel_path, 'w') as f:
                json.dump(symmetry_panel_data, f)


if __name__=="__main__":
    smd_root_folder= '/data/AIGP/style3D_data/dress/objs'
    panel_folder = '/home/gyy/code/garment_pattern_lib/data/annotated/dress/dress02/choices_0'
    exported_symmetry_helper(panel_folder, smd_root_folder)
