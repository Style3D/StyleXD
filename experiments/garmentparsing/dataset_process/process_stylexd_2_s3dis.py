import os
import argparse
import glob
import numpy as np
from tqdm import tqdm

try:
    import open3d
except ImportError:
    import warnings

    warnings.warn("Please install open3d for parsing normal")

try:
    import trimesh
except ImportError:
    import warnings

    warnings.warn("Please install trimesh for parsing normal")

area_mesh_dict = {}

def parse_garment(
        garment, angle, dataset_root, output_root, align_angle=True, parse_normal=False
):
    print("Parsing: {}".format(garment))
    # todo:修改
    classes = [
        "bodyback",
        "bodyfront",
        "skirtback",
        "skirtfront",
        "sleeve",
        "shoulder",
        "belt",
        "cuff",
        "collar",
        "skirtside",
        "ruffles",
        "hem",
        "hat",
        "bodyside",
        "AnotherPanel"
    ]
    class2label = {cls: i for i, cls in enumerate(classes)}

    # 确定房间点云数据和保存路径
    source_dir = os.path.join(dataset_root, garment)
    save_path = os.path.join(output_root, garment)
    os.makedirs(save_path, exist_ok=True)

    # 获取房间中每个物体的点云文件路径
    Panel_path_list = sorted(glob.glob(os.path.join(source_dir, "Annotations/*.txt")))

    # 坐标、颜色、法线、、GT实例分割
    garment_coords = []
    garment_colors = []
    garment_normals = []
    garment_semantic_gt = []  # GT semantic segmrntation
    garment_instance_gt = []  # Gt instance segmentation

    for Panel_id, panel_path in enumerate(Panel_path_list):
        panel_name = os.path.basename(panel_path).split("_")[0]
        panel = np.loadtxt(panel_path)
        coords = panel[:, :3]
        normals = panel[:, 3:6]
        colors = np.array([70, 70, 70])  # use same color
        colors = np.repeat(colors, coords.shape[0])
        colors = colors.reshape([-1, 3])

        class_name = panel_name if panel_name in classes else "AnotherPanel"

        semantic_gt = np.repeat(class2label[class_name], coords.shape[0])
        semantic_gt = semantic_gt.reshape([-1, 1])
        instance_gt = np.repeat(Panel_id, coords.shape[0])
        instance_gt = instance_gt.reshape([-1, 1])

        garment_coords.append(coords)
        garment_colors.append(colors)
        garment_semantic_gt.append(semantic_gt)
        garment_instance_gt.append(instance_gt)
        garment_normals.append(normals)

    garment_coords = np.ascontiguousarray(np.vstack(garment_coords))
    garment_colors = np.ascontiguousarray(np.vstack(garment_colors))
    garment_semantic_gt = np.ascontiguousarray(np.vstack(garment_semantic_gt))
    garment_instance_gt = np.ascontiguousarray(np.vstack(garment_instance_gt))
    garment_normals = np.ascontiguousarray(np.vstack(garment_normals))

    np.save(os.path.join(save_path, "coord.npy"), garment_coords.astype(np.float32))
    np.save(os.path.join(save_path, "color.npy"), garment_colors.astype(np.uint8))
    np.save(os.path.join(save_path, "segment.npy"), garment_semantic_gt.astype(np.int16))
    np.save(os.path.join(save_path, "instance.npy"), garment_instance_gt.astype(np.int16))
    np.save(os.path.join(save_path, "normal.npy"), garment_normals.astype(np.float32))


def main_process():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--splits",
        required=True,
        nargs="+",
        choices=["Area_1", "Area_2", "Area_3", "Area_4", "Area_5", "Area_6"],
        help="Splits need to process ([Area_1, Area_2, Area_3, Area_4, Area_5, Area_6]).",
    )
    parser.add_argument(
        "--dataset_root", required=True, help="Path to Stanford3dDataset_v1.2 dataset"
    )
    parser.add_argument(
        "--output_root",
        required=True,
        help="Output path where area folders will be located",
    )
    parser.add_argument(
        "--raw_root",
        default=None,
        help="Path to Stanford2d3dDataset_noXYZ dataset (optional)",
    )
    parser.add_argument(
        "--align_angle", action="store_true", help="Whether align room angles"
    )
    parser.add_argument(
        "--parse_normal", action="store_true", help="Whether process normal"
    )
    parser.add_argument(
        "--num_workers", default=1, type=int, help="Num workers for preprocessing."
    )
    args = parser.parse_args()

    if args.parse_normal:
        assert args.raw_root is not None

    garment_list = []
    angle_list = []

    # Load area information
    print("Loading area information ...")
    print(args.splits)
    for split in args.splits:
        area_info = np.loadtxt(
            os.path.join(
                args.dataset_root,
                split,
                f"{split}_alignmentAngle.txt",
            ),
            dtype=str,
        )

        if area_info.ndim == 1:
            area_info = np.array([area_info])

        garment_list += [os.path.join(split, garment_info[0]) for garment_info in area_info]
        angle_list += [int(room_info[1]) for room_info in area_info]

    print("Processing garments...")
    for garment, angle in tqdm(zip(garment_list, angle_list)):
        parse_garment(
            garment,
            angle,
            args.dataset_root,
            args.output_root,
            args.align_angle,
            args.parse_normal,
        )

if __name__ == "__main__":
    main_process()
