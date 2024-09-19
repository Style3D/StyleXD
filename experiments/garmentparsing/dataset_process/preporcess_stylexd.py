import os
import json
import shutil
import argparse
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_root", required=True,
    )
    parser.add_argument(
        "--output_root",required=True,
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    output_root = args.output_root

    # This setp may take a long time
    if not os.path.exists(output_root):
        print("Copying tree of dataset, this may take a long time")
        shutil.copytree(dataset_root, output_root)

    split = 6
    additional_info = dict(panel_classess=[])

    all_garment_filename = sorted(os.listdir(output_root))
    all_garment_path = [os.path.join(output_root, garment_filename) for garment_filename in all_garment_filename]
    del all_garment_filename

    print("Tranfering Dataset")
    for garment_path in tqdm(all_garment_path):
        garment_name = os.path.basename(garment_path).split('.')[0]
        all_panel_name = sorted(os.listdir(garment_path))

        annotation_path = os.path.join(garment_path, "Annotations")
        os.makedirs(annotation_path, exist_ok=True)

        with open(os.path.join(garment_path, f"{garment_name}.txt"), 'w', encoding="utf-8") as outfile:
            for panel_name in all_panel_name:
                panel_path = os.path.join(garment_path, panel_name)

                if panel_name.split('.')[0].split('_')[0] not in additional_info["panel_classess"]:
                    additional_info["panel_classess"].append(panel_name.split('.')[0].split('_')[0])

                with open(os.path.join(garment_path, panel_path), 'r', encoding="utf-8") as infile:
                    lines = infile.readlines()[1:]
                    outfile.writelines(lines)
                with open(os.path.join(garment_path, panel_path), 'w', encoding="utf-8") as infile:
                    infile.writelines(lines)

                shutil.move(panel_path, os.path.join(annotation_path, panel_name))

    for i in range(1, split + 1):
        os.makedirs(os.path.join(output_root, f"Area_{i}"), exist_ok=True)
        with open(os.path.join(output_root, f"Area_{i}", f"Area_{i}_alignmentAngle.txt"), "w", encoding="utf-8") as f:
            f.writelines([f"## Global alignment angle per disjoint space in Area_{i} ##\n",
                          "## Disjoint Space Name Global Alignment Angle ##"])

    print("Spliting Dataset")
    for idx, garment_path in tqdm(enumerate(all_garment_path)):
        garment_name = os.path.basename(garment_path).split('.')[0]
        Area_idx = int((idx / len(all_garment_path)) / (1 / split)) + 1
        shutil.move(garment_path, os.path.join(output_root, f"Area_{Area_idx}", garment_name))
        AreaAlignmentAngle = os.path.join(output_root, f"Area_{Area_idx}", f"Area_{Area_idx}_alignmentAngle.txt")
        with open(os.path.join(output_root, f"Area_{Area_idx}", f"Area_{Area_idx}_alignmentAngle.txt"), "a",
                  encoding="utf-8") as f:
            f.writelines(f"\n{garment_name} 0")

    additional_info["panel_classess"].append("AnotherPanel")
    with open(os.path.join(output_root, "additional_info.json"), 'w', encoding='utf-8') as f:
        json.dump(additional_info, f, indent=4)
