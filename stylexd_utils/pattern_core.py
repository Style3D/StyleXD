# -*- coding: utf-8 -*-
"""Helper functions for processing pattern.json used in the AIGP project.
"""

import json
import copy

import numpy as np

from matplotlib import pyplot as plt

from typing import List, Dict, Any

from geomdl import fitting, BSpline, utilities

from scipy.spatial import distance_matrix

from .pattern_helper import pattern_to_image 

from .obj_helper import read_obj


_NECKLINES = {
    "": None
}


_STYLETAGS = {
    "length": ["短款", "中长款", "长款"],
    "fitness": ["修身", "合体", "宽松"],
    "sleeve": ["一片袖", "两片袖", "泡泡袖", "连肩袖", "插肩袖", "落肩袖", "吊带"],
    "collar": ["立领", "圆领", "V领", "异形领", "翻领", "交叠领", "平驳领", "戗驳领", "青果领", "飘带领", "烟囱领", "连帽领", "门襟领"],
    "sleeve_length": ["无袖", "短袖", "五分袖", "七分袖", "长袖", "超长袖"]
}


_PATTERN_CATEGORIES = [
    "head", "neck", "shoulder", "arm", "wrist",
    "bodyfront", "bodyback", "bodyside", "hem",
    "waist", "pelvisfront", "pelvisback", "pelvisside", "ankle",
    "pocket", "none"  
]


class Edge():

    def __init__(self, **edge_spec) -> None:
        self.__dict__.update(edge_spec)
        self.next = None
        self.parent = None      # belonging panel object
        print(dir(self))

    def split(self, param=1.0):
        edges = []
        return edges
    
    def merge(self, edges):
        pass

    def serialize(self) -> Dict:
        return self.__dict__

    def vectorize(self) -> np.ndarray:
        pass


class Panel:
    def __init__(self, **panel_spec) -> None:
        self.__dict__.update(panel_spec)

    def serialize(self) -> Dict:
        return self.__dict__
    
    def vectorize(self) -> np.ndarray:
        pass

    def visualize(self, output_file:str=None, show:bool=False):
        pass


class PanelGroups:
    def __init__(self, panels, group_id) -> None:
        self.panels = panels
        self.group_id = group_id
        self.center = np.mean(np.array([x.center for x in self.panels]), axis=0)

    def optimize(self) -> None:
        ''' Optimize panel arrangement. '''
        pass

    def visualize(self) -> None:
        pass


class BasicPattern(object):
    def __init__(self, pattern_file) -> None:
        self.spec_file = pattern_file 
        self.coord_system = 'relative'  
        self.tag = None     
        self.reload_json(self.spec_file)


    def _convert_edge_abs_to_rel(self, edge_spec: Dict, verts: np.ndarray) -> Dict:
        edge_crtl_pts = np.asarray(edge_spec['controlPoints'])[:, :2]
        edge_bezier_pts = np.asarray(edge_spec['bezierPoints'])
        
        if np.any(edge_bezier_pts):
            edge_bezier_pts = edge_bezier_pts[:, :2] + edge_crtl_pts    # convert from vector repr. to point repr.

        # Convert ctrl points to relative coordinate
        start = edge_crtl_pts[0, :]
        end = edge_crtl_pts[-1, :] 

        start_idx = np.argmin(np.linalg.norm(verts[:, :2] - start[None], axis=1))
        end_idx = np.argmin(np.linalg.norm(verts[:, :2] - end[None], axis=1))
        edge_spec["endPoints"] = [int(start_idx), int(end_idx)]

        # convert edge parameters    
        edge = end - start
        edge_len = np.linalg.norm(edge)

        def project_point(pt):
            converted = [None, None]
            point_vec = pt - start
            projected_len = edge.dot(point_vec) / edge_len
            converted[0] = projected_len / edge_len

            control_projected = edge * converted[0]
            vert_comp = point_vec - control_projected
            converted[1] = np.linalg.norm(vert_comp) / edge_len
            converted[1] *= -np.sign(np.cross(point_vec, edge)) 

            return converted

        converted_ctrl_pts = np.array([project_point(x) for x in edge_crtl_pts], dtype=np.float32)
        converted_ctrl_pts = np.concatenate([converted_ctrl_pts, np.zeros_like(converted_ctrl_pts[:, :1])], axis=-1)
        edge_spec['controlPoints'] = converted_ctrl_pts.tolist()

        if np.any(edge_bezier_pts):
            converted_bezier_pts = np.array([project_point(x) for x in edge_bezier_pts], dtype=np.float32)
            converted_bezier_pts = np.concatenate([converted_bezier_pts, np.zeros_like(converted_bezier_pts[:, :1])], axis=-1)
            edge_spec['bezierPoints'] = converted_bezier_pts.tolist()

        return edge_spec


    def _convert_edge_rel_to_abs(self, edge_spec: Dict, verts: np.ndarray) -> Dict:
        start = verts[edge_spec["endPoints"][0], :2]
        end = verts[edge_spec["endPoints"][1], :2]

        edge = end - start                      
        edge_perp = np.array([-edge[1], edge[0]])    # Y-axis, perpendicular to edge

        def unproject_point(pt):
            conv_start = start + pt[0] * edge
            conv_point = conv_start + pt[1] * edge_perp
            return conv_point

        edge_crtl_pts = np.array(edge_spec['controlPoints'])[:, :2]
        edge_bezier_pts = np.array(edge_spec['bezierPoints'])[:, :2]

        converted_ctrl_pts = np.array([unproject_point(x) for x in edge_crtl_pts], dtype=np.float32)
        converted_ctrl_pts = np.concatenate([converted_ctrl_pts, np.zeros_like(converted_ctrl_pts[:, :1])], axis=-1)
        edge_spec['controlPoints'] = converted_ctrl_pts.tolist()

        if np.any(edge_bezier_pts):
            converted_bezier_pts = np.array([unproject_point(x) for x in edge_bezier_pts], dtype=np.float32)
            converted_bezier_pts = np.concatenate([converted_bezier_pts, np.zeros_like(converted_bezier_pts[:, :1])], axis=-1)
            edge_spec['bezierPoints'] = (converted_bezier_pts-converted_ctrl_pts).tolist()

        del edge_spec["endPoints"]

        return edge_spec


    def convert_coord_abs_to_rel(self):
        assert len(self.panels) > 0, 'Empty pattern'
        for panel in self.panels:
            verts = np.array(panel['vertices'])
            for e_idx, edge_spec in enumerate(panel['edges']):
                panel['edges'][e_idx] = self._convert_edge_abs_to_rel(edge_spec, verts)
        self.coord_system = 'relative'


    def convert_coord_rel_to_abs(self):
        for panel in self.panels:
            verts = np.array(panel['vertices'])
            for e_idx, edge_spec in enumerate(panel['edges']):
                panel['edges'][e_idx] = self._convert_edge_rel_to_abs(edge_spec, verts)
        self.coord_system = 'absolute'


    def reload_json(self, pattern_file):
        with open(pattern_file, 'r') as f: pattern_data = json.load(f)
        coord_system = pattern_data.get('coordinate', 'absolute')
        self.panels = pattern_data['panels']
        self.stitches = pattern_data['stitches']

            
    def serialize(self, path, coord='absolute', tag=None, empty_ok=False):
        if not empty_ok and not self.pattern['panels']:
            raise ValueError('Empty pattern')

        if coord == 'absolute' and self.coord_system == 'relative':
            self._convert_coord_rel_to_abs()
        elif coord == 'relative' and self.coord_system == 'absolute':
            self._convert_coord_abs_to_rel()

        if self.tag is None and tag is not None:
            self.pattern['tag'] = tag

        with open(path, 'w') as f:
            json.dump(self.pattern, f, indent=4, ensure_ascii=False)

    

    

def generate_pattern(template:Dict, style_tags:str=None) -> Dict:
    ''' Generate a pattern from a template and panel data.
    Input:
        template: Dict, pattern template
        panel_data: Dict, panel data
        panel_ids: List[int], panel ids to merge
        style_tags: List[str], style tags
    Output:
        pattern: Dict, generated pattern
    '''

    style_tags = style_tags.split('|')

    pass

def rearrange_pattern(pattern:Dict, arrangement:Dict) -> Dict:
    pass


def parse_style_tags(style_tags:str) -> List[int]:
    
    pass


def generate_style_tag(panel_data:Dict):
    panel_labels = [x['label'].split('|')[0] for x in panel_data['panels']]
    pass


def merge_panels(panel_data:Dict, panel_ids:List[int]):
    ''' Merge two panels in a garment with stitching info.
    Input:
        panel_data: Dict, panel data
        panel_ids: List[int], panel ids to merge
    Output:
        panel_data: Dict, modified panel data
    '''    
    pass


def modify_skirt_length(panel_data:Dict, min:float, max:float):
    pass


def modify_sleeve_length(panel_data:Dict, min:float, max:float):
    pass


def modify_collar(panel_data:Dict, new_collar: Dict):
    '''
    Input:
        panel_data: Dict, panel data
        new_collar: Dict, new collar as list of edges
    Output:
        panel_data: Dict, modified panel data
    '''
    pass


def modify_neckline(panel_data:Dict, new_neckline: Dict):
    '''
    Input:
        panel_data: Dict, panel data
        new_neckline: Dict, new neckline as list of edges
    Output:
        panel_data: Dict, modified panel data
    '''
    pass


def open_dart(panel_data:Dict, edge_id:int, num_darts, open_length:float):
    ''' Open num_darts darts on the edge with edge_id, each dart has open_length 
    Input:
        panel_data: Dict, panel data
        new_neckline: Dict, new neckline as list of edges
    Output:
        panel_data: Dict, modified panel data
    '''
    pass


def add_dart(panel_data: Dict, panle_id, vert_0, vert_1):
    pass


if __name__ == "__main__":
    import os
    import time
    import json
    from glob import glob
    import random
    from tqdm import tqdm
    import argparse
    import shutil

    parser = argparse.ArgumentParser(
        description="Test pattern processing lib."
    )
    parser.add_argument(
        "-i", "--input", default=".\examples", type=str, help="Input directory including patterns and (optional) mesh objs."
    )
    parser.add_argument(
        "-o", "--output", default=".\outputs", type=str, help="Output directory, output pattern.json with lables, arrangements and stitches."
    )

    args, cfg_cmd = parser.parse_known_args()

    all_items = glob(f'{args.input}/*/pattern.json', recursive=True)
    print('Total number of items: ', len(all_items))

    os.makedirs(os.path.join(args.output, 'failed'), exist_ok=True)
    print('Creating output directory: ', args.output)

    succeed_cnt = 0

    for data_item in tqdm(sorted(all_items)):
        try:
            out_item_fp = os.path.join(args.output, os.path.basename(os.path.dirname(data_item)) + '.json')
            os.makedirs(os.path.dirname(out_item_fp), exist_ok=True)

            obj_id = os.path.basename(os.path.dirname(data_item))
            obj_fp = os.path.join(os.path.dirname(data_item), f'{obj_id}.obj')

            mesh_obj = read_obj(obj_fp)
            avatar_fp = os.path.join(os.path.dirname(data_item), 'avatar.json')
            with open(avatar_fp, 'r') as f: avatar_data = json.load(f)        

            pattern_to_image(data_item, out_item_fp, mesh_obj=mesh_obj, avatar=avatar_data)

            succeed_cnt += 1
        
        except Exception as e:
            print('Error: ', data_item, e)
            shutil.copyfile(data_item, os.path.join(args.output, 'failed', os.path.basename(data_item)))
            continue

    print('[DONE] processing pattern.json: ', succeed_cnt, len(all_items))
