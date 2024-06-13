import json
from tqdm import tqdm
import csv
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import requests
import argparse

from file_helper import save_file
from constants import PROJECT_LIST


def get_exported_project_json(project_id_lst: list[int], output_folder: str, info: dict):
    results = []
    os.makedirs(output_folder, exist_ok=True)
    print(f'[INFO]Start exporting project annotation results...')

    token = info["labelstudio"]["key"]

    for project_id in tqdm(project_id_lst):
        url = f"http://192.168.28.166:8081/api/projects/{project_id}/export?exportType=JSON"
        headers = {
            'Authorization': f'Token {token}'
        }

        # 发送请求
        # `verify=False` 用于忽略SSL证书验证
        response = requests.get(url, headers=headers, verify=False)

        # 将结果写入文件
        if response.status_code == 200:
            filename = os.path.join(
                output_folder, f'project_{project_id}_anno_res.json')
            with open(filename, 'wb') as file:
                file.write(response.content)
            results.append(filename)
        else:
            print("请求失败，状态码：", response.status_code)
    return results


def read_json(json_path: str):
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    return json_data


def save_json(json_content, output_path):
    with open(output_path, 'w') as f:
        json.dump(json_content, f)


def time_formatter(time_str: str):
    # origin: 2023-10-30T05:51:59.235486Z
    # result: [2023-10-30, 13:51:59]
    date, time = time_str.split('T')[0], time_str.split('T')[1].split('.')[0]
    time = str(int(time[:2]) + 8) + time[2:]
    return [date, time]


def create_hourly_charts_for_each_annotater(csv_file, outfolder):
    os.makedirs(outfolder, exist_ok=True)
    df = pd.read_csv(csv_file)
    user_id = [4, 5, 11, 12]
    # 数据预处理
    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    df['hour'] = df['datetime'].dt.hour  # 截取小时
    df['date'] = df['datetime'].dt.date

    # 数据分组和聚合
    grouped = df.groupby(['annotater', 'date', 'hour']
                         ).size().reset_index(name='count')

    # 为每个annotater的每一天创建图表
    for annotater in grouped['annotater'].unique():
        if annotater in user_id:
            annotater_group = grouped[grouped['annotater'] == annotater]

            annotater_folder = f'{outfolder}/{annotater}_charts'
            if not os.path.exists(annotater_folder):
                os.makedirs(annotater_folder)

            for date in annotater_group['date'].unique():
                daily_data = annotater_group[annotater_group['date'] == date]

                plt.figure()
                bars = plt.bar(
                    daily_data['hour'], daily_data['count'], width=0.8)  # 设置柱的宽度
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval,
                             int(yval), ha='center', va='bottom')
                plt.xlabel('Hour')
                plt.ylabel('Data Count')
                plt.title(
                    f'Data Count per Hour for Annotater {annotater} on {date}')
                plt.xticks(range(8, 23))  # 设置x轴标记为8点到22点
                plt.tight_layout()

                # 保存图表
                plt.savefig(os.path.join(annotater_folder, f'{date}.png'))
                plt.close()


def calculate_drop_ratio(csv_file, output_file):
    df = pd.read_csv(csv_file)

    # 计算每个project_id的choices=3的比例
    drop_ratio = df[df['choices'] == 3].groupby(
        'project_id').size() / df.groupby('project_id').size()

    drop_result = [
        f"{key} {np.round(value, 5)}" for key, value in drop_ratio.to_dict().items()]

    save_file(drop_result, output_file)
    print(f'[INFO]Save drop rate of each project in {output_file}.')


def get_task_info(task):
    annotater = task['annotations'][0]['completed_by']

    project_id = task['project']

    task_id = task['id']

    date, time = time_formatter(task['updated_at'])

    anno_result = task['annotations'][0]['result']
    choices = None
    for submodule in anno_result:
        if submodule['type'] == 'choices':
            choices = submodule['value']['choices'][0]

    panels_uuid = task['meta']['id']

    return [annotater, date, time, choices, panels_uuid, project_id, task_id]


def efficient_summary(exported_path_list: list[list]):
    print("[INFO]Start calculating efficient summary...")
    final_res = []
    for exported_path in tqdm(exported_path_list):
        exported_data = read_json(exported_path)
        for task in exported_data:
            anno_list = get_task_info(task)
            final_res.append(anno_list)
    return final_res


def save_csv(res_content, output_path):
    header = ['annotater', 'date', 'time', 'choices',
              'panels_uuid', 'project_id', 'task_id']
    with open(output_path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(res_content)


def main(csv_outpth, exported_json_folder, chart_folder, infofile):
    # 0. get info.json
    cfg_data = json.load(open(infofile, "rb"))

    # 1. export annotation result from label studio
    exported_json_file_list = get_exported_project_json(
        PROJECT_LIST, output_folder=exported_json_folder, info=cfg_data)

    # 2. gather anno infomations from exported json
    efficient_summary_result = efficient_summary(exported_json_file_list)
    save_csv(efficient_summary_result, csv_outpth)

    # 3. create charts to show anno efficiency
    create_hourly_charts_for_each_annotater(csv_outpth, outfolder=chart_folder)

    # 4. calculate drop ratio for each project
    calculate_drop_ratio(csv_outpth, csv_outpth.replace('.csv', '_drop_rate.txt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--info_file", type=str, default='../data/info.json')
    parser.add_argument("-o", "--outfolder", type=str)

    args, cfg_cmd = parser.parse_known_args()
    if not args.outfolder:
        args.outfolder = '/home/gyy/code/garment_pattern_lib/data/label_studio_anno_export/exported_from_code'
    os.makedirs(args.outfolder, exist_ok=True)
    chart_folder = f'{args.outfolder}/effi_chart'
    final_anno_csv = f'{args.outfolder}/annotime_analyse_summary.csv'

    main(final_anno_csv, args.outfolder, chart_folder, args.info_file)
