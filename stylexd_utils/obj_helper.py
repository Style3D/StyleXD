from __future__ import annotations

import sys

import numpy as np
from numpy.typing import ArrayLike

from typing import Dict


class CellBlock:
    def __init__(
        self,
        cell_type: str,
        data: list | np.ndarray,
        tags: list[str] | None = None,
    ):
        self.type = cell_type
        self.data = data

        if cell_type.startswith("polyhedron"):
            self.dim = 3
        else:
            self.data = np.asarray(self.data)
            self.dim = 2

        self.tags = [] if tags is None else tags

    def __repr__(self):
        items = [
            "meshio CellBlock",
            f"type: {self.type}",
            f"num cells: {len(self.data)}",
            f"tags: {self.tags}",
        ]
        return "<" + ", ".join(items) + ">"

    def __len__(self):
        return len(self.data)


class Mesh:
    def __init__(
        self,
        points: ArrayLike,
        cells: dict[str, ArrayLike] | list[tuple[str, ArrayLike] | CellBlock],
        point_data: dict[str, ArrayLike] | None = None,
        cell_data: dict[str, list[ArrayLike]] | None = None,
        field_data=None,
        point_sets: dict[str, ArrayLike] | None = None,
        cell_sets: dict[str, list[ArrayLike]] | None = None,
        info=None,
    ):
        self.points = np.asarray(points)
        if isinstance(cells, dict):
            cells = list(cells.items())

        self.cells = []
        for cell_block in cells:
            if isinstance(cell_block, tuple):
                cell_type, data = cell_block
                cell_block = CellBlock(
                    cell_type,
                    # polyhedron data cannot be converted to numpy arrays
                    # because the sublists don't all have the same length
                    data if cell_type.startswith("polyhedron") else np.asarray(data),
                )
            self.cells.append(cell_block)

        self.point_data = {} if point_data is None else point_data
        self.cell_data = {} if cell_data is None else cell_data
        self.field_data = {} if field_data is None else field_data
        self.point_sets = {} if point_sets is None else point_sets
        self.cell_sets = {} if cell_sets is None else cell_sets
        self.info = info

        # print('*** field data: ', self.field_data)

        # assert point data consistency and convert to numpy arrays
        for key, item in self.point_data.items():
            self.point_data[key] = np.asarray(item)
            if len(self.point_data[key]) != len(self.points):
                raise ValueError(
                    f"len(points) = {len(self.points)}, "
                    f'but len(point_data["{key}"]) = {len(self.point_data[key])}'
                )

        # assert cell data consistency and convert to numpy arrays
        for key, data in self.cell_data.items():
            if len(data) != len(cells):
                raise ValueError(
                    f"Incompatible cell data '{key}'. "
                    f"{len(cells)} cell blocks, but '{key}' has {len(data)} blocks."
                )

            for k in range(len(data)):
                data[k] = np.asarray(data[k])
                if len(data[k]) != len(self.cells[k]):
                    raise ValueError(
                        "Incompatible cell data. "
                        + f"Cell block {k} ('{self.cells[k].type}') "
                        + f"has length {len(self.cells[k])}, but "
                        + f"corresponding cell data item has length {len(data[k])}."
                    )


def __read_buffer(f):
    points = []
    vertex_normals = []
    texture_coords = []

    face_groups = []  # face index
    face_group_tags = []  # face group tag (str)
    face_group_ids = []  # custom face group id (int)
    face_group_id = -1  # tmp value to store face group id

    while True:
        line = f.readline()

        if not line:
            break

        try:
            line = line.decode()
        except:
            continue

        strip = line.strip()

        if len(strip) == 0 or strip[0] == "#":
            continue

        split = strip.split()

        if split[0] == "v":
            points.append([float(item) for item in split[1:]])
        elif split[0] == "vn":
            vertex_normals.append([float(item) for item in split[1:]])
        elif split[0] == "vt":
            texture_coords.append([float(item) for item in split[1:]])
        elif split[0] == "s":
            # "s 1" or "s off" controls smooth shading
            pass
        elif split[0] == "f":
            dat = [int(item.split("/")[0]) for item in split[1:]]
            if len(face_groups) == 0 or (
                len(face_groups[-1]) > 0 and len(face_groups[-1][-1]) != len(dat)
            ):
                face_groups.append([])
                face_group_ids.append([])

            face_groups[-1].append(dat)
            face_group_ids[-1].append(face_group_id)
        elif split[0] == "g":
            # new group
            face_groups.append([])
            face_group_ids.append([])
            face_group_id += 1
            face_group_tags.append(split[1].strip() if len(split) > 1 else "")
        else:
            # who knows
            pass

    # remove empty groups
    assert len(face_groups) == len(face_group_ids) and len(face_groups) == len(
        face_group_tags
    ), "Number of face groups mismatch."

    valid_index = [len(x) > 0 for x in face_groups]
    face_groups = [face_groups[i] for i in range(len(face_groups)) if valid_index[i]]
    face_group_ids = [
        face_group_ids[i] for i in range(len(face_group_ids)) if valid_index[i]
    ]
    face_group_tags = [
        face_group_tags[i] for i in range(len(face_group_tags)) if valid_index[i]
    ]

    points = np.array(points)
    texture_coords = np.array(texture_coords)
    vertex_normals = np.array(vertex_normals)
    point_data = {}
    if len(texture_coords) > 0:
        point_data["obj:vt"] = texture_coords
    if len(vertex_normals) > 0:
        point_data["obj:vn"] = vertex_normals

    # convert to numpy arrays
    face_groups = [np.array(f) for f in face_groups]
    cell_data = {"obj:group_ids": []}
    cells = []
    for f, gid in zip(face_groups, face_group_ids):
        if f.shape[1] == 3:
            cells.append(CellBlock("triangle", f - 1))
        elif f.shape[1] == 4:
            cells.append(CellBlock("quad", f - 1))
        else:
            cells.append(CellBlock("polygon", f - 1))
        cell_data["obj:group_ids"].append(gid)

    # logging field data
    field_data = {"obj:group_tags": face_group_tags}

    return Mesh(
        points, cells, point_data=point_data, cell_data=cell_data, field_data=field_data
    )


def read_obj(filename: str) -> Dict:
    with open(filename, "rb") as f:
        return __read_buffer(f)


def write_obj(filename, mesh):
    raise NotImplementedError
    # for c in mesh.cells:
    #     if c.type not in ["triangle", "quad", "polygon"]:
    #         raise WriteError(
    #             "Wavefront .obj files can only contain triangle or quad cells."
    #         )

    # with open_file(filename, "w") as f:
    #     f.write(
    #         "# Created by meshio v{}, {}\n".format(
    #             __version__, datetime.datetime.now().isoformat()
    #         )
    #     )
    #     for p in mesh.points:
    #         f.write(f"v {p[0]} {p[1]} {p[2]}\n")

    #     if "obj:vn" in mesh.point_data:
    #         dat = mesh.point_data["obj:vn"]
    #         fmt = "vn " + " ".join(["{}"] * dat.shape[1]) + "\n"
    #         for vn in dat:
    #             f.write(fmt.format(*vn))

    #     if "obj:vt" in mesh.point_data:
    #         dat = mesh.point_data["obj:vt"]
    #         fmt = "vt " + " ".join(["{}"] * dat.shape[1]) + "\n"
    #         for vt in dat:
    #             f.write(fmt.format(*vt))

    #     for cell_block in mesh.cells:
    #         fmt = "f " + " ".join(["{}"] * cell_block.data.shape[1]) + "\n"
    #         for c in cell_block.data:
    #             f.write(fmt.format(*(c + 1)))


def __vis_mesh_o3d(mesh, cmap):
    if "o3d" not in sys.modules:
        import open3d as o3d

    clothPieces = []
    for idx, cell in enumerate(mesh.cells):
        clothPiece = o3d.geometry.TriangleMesh()
        clothPiece.vertices = o3d.utility.Vector3dVector(mesh.points)
        clothPiece.triangles = o3d.utility.Vector3iVector(cell.data)

        clothPiece.paint_uniform_color(cmap[idx % len(cmap)])
        clothPieces.append(clothPiece)

    o3d.visualization.draw_geometries(clothPieces)


def __vis_mesh_plotly(mesh, cmap, landmarks=None, lm_cmap=None):
    raise NotImplementedError("TBD visualization with plotly.")
