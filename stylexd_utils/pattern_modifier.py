# -*- coding: utf-8 -*-
"""Helper functions for processing pattern.json used in the AIGP project.
"""

import json
import copy

import numpy as np
from matplotlib import pyplot as plt
from typing import List, Dict
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
