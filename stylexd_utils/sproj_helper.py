import os
import argparse

from glob import glob

import time
import logging

import zipfile
import shutil
import subprocess
import datetime

from utils.ding import ding_msg, markdown_msg, text_msg

from multiprocessing import Pool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_USER = "XXX"  # ding message @user


def __export_joints(input_file, output_file):
    # tmp_dir = os.path.join(os.path.dirname(output_file), "tmp")
    # os.makedirs(tmp_dir, exist_ok=True)

    # assert os.path.isfile(input_file) and input_file.split(".")[1] == "sproj", (
    #     "[ERROR] Input file not exist: %s" % (input_file)
    # )

    # tmp_file = os.path.join(
    #     tmp_dir, os.path.basename(input_file).replace(".sproj", ".zip")
    # )
    # shutil.copyfile(input_file, tmp_file)
    # with zipfile.ZipFile(tmp_file, "r") as zip_ref:
    #     zip_ref.extractall(tmp_file.replace(".zip", ""))
    # os.remove(tmp_file)

    # # open project file
    # with open(os.path.join(tmp_file.replace(".zip", ""), "project.json"), "r") as f:
    #     sproj_data = json.load(f)
    #     cloud_items = dict(
    #         [(x["uuid"]["uuid"], x) for x in sproj_data["objects"] if "uuid" in x]
    #     )
    #     print(cloud_items)
    raise NotImplementedError



def __export_obj(input_file, output_file, worker, export_preview=False):
    tic = time.perf_counter()

    # clean up cache
    try:
        command = [
            worker,
            "--aiExport",
            os.path.abspath(output_file),
            os.path.abspath(input_file),
        ]
        subprocess.run(command, input=b"\n", stderr=subprocess.STDOUT, timeout=60)

        output_zip_file = output_file.replace(".obj", ".zip")
        with zipfile.ZipFile(output_zip_file, "r") as zip_ref:
            zip_ref.extractall(output_zip_file.replace(".zip", ""))
        os.remove(output_zip_file)
        tex_imgs = glob(os.path.join(output_zip_file.replace(".zip", ""), "*.jpg"))
        for tex_img_fp in tex_imgs:
            os.remove(tex_img_fp)

        # export snapshot
        if export_preview:
            sproj_zip_file = os.path.join(
                os.path.dirname(output_file),
                os.path.basename(input_file).replace(".sproj", "_sproj.zip"),
            )

            shutil.copyfile(input_file, sproj_zip_file)
            with zipfile.ZipFile(sproj_zip_file, "r") as zip_ref:
                zip_ref.extractall(sproj_zip_file.replace(".zip", ""))
                front_snapshot = glob(
                    os.path.join(
                        sproj_zip_file.replace(".zip", ""), "Project*_front.png"
                    )
                )
                if len(front_snapshot):
                    front_snapshot = front_snapshot[0]
                    shutil.copyfile(
                        front_snapshot,
                        os.path.join(output_file.replace(".obj", ""), "front.png"),
                    )
            os.remove(sproj_zip_file)
            shutil.rmtree(sproj_zip_file.replace(".zip", ""), ignore_errors=True)
        
        toc = time.perf_counter()
        logger.info("[SUCCEED] Processing time for %s: %.4f" % (input_file, toc - tic))

        return True

    except Exception as e:
        logger.info("[ERROR] Processing %s: %s" % (input_file, str(e)))
        return False


def __export_panels(input_file, output_file, worker):
    tic = time.perf_counter()

    try:
        command = [worker, '--patExport', os.path.abspath(output_file), os.path.abspath(input_file)]
        subprocess.run(command, input=b'\n', stderr=subprocess.STDOUT, timeout=60)

        with zipfile.ZipFile(output_file, 'r') as zip_ref:
            zip_ref.extractall(output_file.replace(".zip", ""))
            shutil.move(os.path.join(
                output_file.replace(".zip", ""), 'pattern.json'),
                output_file.replace(".zip", ".json"))
        
        os.remove(output_file)            
        shutil.rmtree(output_file.replace(".zip", ""), ignore_errors=True)
                    
        toc = time.perf_counter()
        logger.info('[SUCCEED] Processing time for %s: %.4f'%(input_file, toc-tic))

        return True
    
    except Exception as e:
        logger.info('[ERROR] Processing %s: %s'%(input_file, str(e)))
        return False


def __export_sil(input_file, output_file, worker, export_preview=False):
    tic = time.perf_counter()

    # clean up cache
    try:
        command = [
            worker,
            "--silExport",
            os.path.abspath(output_file),
            os.path.abspath(input_file),
        ]
        subprocess.run(command, input=b"\n", stderr=subprocess.STDOUT, timeout=300)
        
        toc = time.perf_counter()
        logger.info("[SUCCEED] Processing time for %s: %.4f" % (input_file, toc - tic))

        return True

    except Exception as e:
        logger.info("[ERROR] Processing %s" % (input_file))
        msg = text_msg("[ERROR] " + input_file + ":\t" + str(e))
        ding_msg(msg)
        return False


def __export_smd(input_file, output_file, worker):
    tic = time.perf_counter()

    try:
        command = [worker, '--smd', os.path.abspath(output_file), os.path.abspath(input_file)]
        subprocess.run(command, input=b'\n', stderr=subprocess.STDOUT, timeout=60)

        os.rename(output_file, output_file.replace(".smd", ".zip"))
        output_zip_file = output_file.replace(".smd", ".zip")
        with zipfile.ZipFile(output_zip_file, "r") as zip_ref:
            zip_ref.extractall(output_zip_file.replace(".zip", ""))
            shutil.move(os.path.join(
                output_zip_file.replace(".zip", ""), 'smd.json'),
                output_zip_file.replace(".zip", ".json"))

        os.remove(output_zip_file)
        shutil.rmtree(output_zip_file.replace(".zip", ""), ignore_errors=True)
                    
        toc = time.perf_counter()
        logger.info('[SUCCEED] Processing time for %s: %.4f'%(input_file, toc-tic))

        return True
    
    except Exception as e:
        logger.info('[ERROR] Processing %s: %s'%(input_file, str(e)))
        return False


def __export_preview(input_file, output_file):
    output_dir = os.path.dirname(input_file)

    sproj_zip_file = os.path.join(
        output_dir, os.path.basename(input_file).replace(".sproj", "_sproj.zip")
    )
    shutil.copyfile(input_file, sproj_zip_file)

    try:
        with zipfile.ZipFile(sproj_zip_file, "r") as zip_ref:
            zip_ref.extractall(sproj_zip_file.replace(".zip", ""))
            front_snapshot = glob(
                os.path.join(sproj_zip_file.replace(".zip", ""), "Project*_front.png")
            )
            if len(front_snapshot):
                front_snapshot = front_snapshot[0]
                shutil.copyfile(front_snapshot, output_file)
        os.remove(sproj_zip_file)
        shutil.rmtree(sproj_zip_file.replace(".zip", ""), ignore_errors=True)
    except Exception:
        logger.error("%s\t 0", input_file, exc_info=False)
        if os.path.exists(sproj_zip_file):
            os.remove(sproj_zip_file)
        if os.path.isdir(sproj_zip_file.replace(".zip", "")):
            shutil.rmtree(sproj_zip_file.replace(".zip", ""), ignore_errors=True)
        return


def run_style_cmd(
    input_file, output_dir=None, style_cmd="obj", worker="Style3DTest.exe"
):
    if style_cmd == "obj":
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".sproj", ".obj")
        )
        # __export_obj_old(input_file, output_file, worker)
        res = __export_obj(input_file, output_file, worker)

    elif style_cmd == "smd":
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".sproj", ".smd")
        )
        res = __export_smd(input_file, output_file, worker)

    elif style_cmd == "preview":
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".sproj", ".png")
        )
        res = __export_preview(input_file, output_file)

    elif style_cmd == "sil":
        output_file = os.path.join(
            output_dir, os.path.basename(input_file).replace(".sproj", ".zip")
        )
        res = __export_sil(input_file, output_file, worker)

    elif style_cmd == "pattern":
        output_file = os.path.join(output_dir, os.path.basename(input_file).replace(".sproj", ".zip"))
        res = __export_panels(input_file, output_file, worker)

    else:
        raise ValueError("Invalid export command: %s." % (style_cmd))

    return "%s\t %d\n" % (input_file, res)


def main():
    parser = argparse.ArgumentParser(
        description="Processing *.sproj -> cloth piece obj"
    )
    parser.add_argument(
        "-i",
        "--input",
        default="E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\sprojs",
        type=str,
        help="Input directory.",
    )
    parser.add_argument(
        "-o", "--output", default=None, type=str, help="Output directory."
    )
    parser.add_argument(
        "-e", "--exe", default=None, type=str, help="Path to executable."
    )
    parser.add_argument(
        "-c", "--cmd", default=None, type=str, help="Style3D CLI command."
    )
    parser.add_argument(
        "-r", "--range", default=None, type=str, help="Path to executable."
    )
    parser.add_argument(
        "-p",
        "--pool",
        default=5,
        type=int,
        help="Number of threads for multi processing.",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        default=500,
        type=int,
        help="Number of threads for multi processing.",
    )
    args, _ = parser.parse_known_args()

    ## find all sproj files in input dir
    all_files = sorted(glob(os.path.join(args.input, "*.sproj"), recursive=True))
    if args.output is None:
        args.output = os.path.join(os.path.dirname(args.input), args.cmd + "s")

    if args.range is not None:
        begin, end = args.range.split(",")
        begin, end = max(0, int(begin)), min(int(end), len(all_files))
        all_files = all_files[begin:end]
        logger.info("Extracting range: %d %s" % (len(all_files), args.output))

    os.makedirs(args.output, exist_ok=True)

    ## check duplicate
    if os.path.exists(args.output):
        # check log file (in case some batches has been moved out due to memory issue)
        if os.path.exists(os.path.join(args.output, "app.log")):
            logger.info(
                "Resuming from log file: %s" % os.path.join(args.output, "app.log")
            )
            with open(os.path.join(args.output, "app.log"), "r") as f:
                lines = f.readlines()
                processed = [
                    x.split("\t")[0] for x in lines if x.split("\t")[1].strip() != "0"
                ]
                all_files = [x for x in all_files if x not in processed]

        # check existing folders
        processed = [
            x
            for x in os.listdir(args.output)
            if os.path.exists(os.path.join(args.output, x, "avatar.json"))
            and os.path.exists(os.path.join(args.output, x, x + ".obj"))
        ]
        print("*** processed: ", len(processed))
        all_files = [
            x
            for x in all_files
            if os.path.basename(x).replace(".sproj", "") not in processed
        ]

    else:
        os.makedirs(args.output, exist_ok=True)

    logger.info(
        "Input (%d files): %s\nOutput: %s \n"
        % (len(all_files), args.input, args.output)
    )

    ## change to executable directory
    if args.exe is not None:
        os.chdir(os.path.dirname(args.exe))
    exe_worker = os.path.basename(args.exe) if args.exe is not None else None
    params = [(x, args.output, args.cmd, exe_worker) for x in all_files]

    batch_params = [
        params[i : i + args.batchsize] for i in range(0, len(params), args.batchsize)
    ]

    date_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    init_msg = f"### [{date_str}] Preprocessing .sproj -> {args.cmd}\n---\n"
    init_msg += f"- **Input Dir  :** {args.input}\n"
    init_msg += f"- **Total Items:** {len(all_files)}\n"
    init_msg += f"- **Output Dir :** {args.output}\n"
    init_msg += (
        f"- **Batch Size :** {args.batchsize} ({len(batch_params)} batches in total).\n"
    )
    init_msg += f"- **Num Pools  :** {args.pool}\n"

    init_msg = markdown_msg("Init Run", init_msg)
    ding_msg(init_msg)

    for batch_idx, batch in enumerate(batch_params):
        with Pool(args.pool) as pool:
            batch_res = pool.starmap(run_style_cmd, batch)

        batch_res_cnt = [int(x.split()[1].strip()) for x in batch_res]

        with open(os.path.join(args.output, "app.log"), "a") as f:
            f.writelines(batch_res)

        msg = markdown_msg(
            "[%03d / %03d] finish batch.",
            "[%03d / %03d] batch info: total %d; succeed %d; failed %d."
            % (
                batch_idx,
                len(batch_params),
                len(batch),
                sum(batch_res_cnt),
                len(batch) - sum(batch_res_cnt),
            ),
        )

        logger.info(">>> Send dingtalk message: %s" % (ding_msg(msg)))
        time.sleep(60)  # sleep 1 min to avoid style3D crash (failed to init OpenGL)

    # Flitering valid objects
    processed_items = glob(os.path.join(args.output, "**", "*/"))
    logger.info("Processed items: %d" % (len(processed_items)))
    valid_cnt = 0
    for item_dir in processed_items:
        item_id = os.path.basename(item_dir[:-1])
        obj_fp = os.path.join(item_dir, item_id + ".obj")
        avatar_fp = os.path.join(item_dir, "avatar.json")
        smd_fp = os.path.join(item_dir, "smd.json")

        if item_id == "bed799d04ad8010d8c926ce4d274d6f4":
            print(item_dir)

        is_valid = (
            os.path.exists(obj_fp)
            and os.path.exists(avatar_fp)
            and os.path.exists(smd_fp)
        )
        valid_cnt += int(is_valid)
        if not is_valid:
            shutil.rmtree(item_dir)

    logger.info(
        "Validate Result: total %d, valid %d, invalid %d"
        % (len(processed_items), valid_cnt, len(processed_items) - valid_cnt)
    )

    date_str = (datetime.datetime.now() + datetime.timedelta(hours=8)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    finilize_msg = f"### [{date_str}] Preprocessing finished.\n---\n"
    finilize_msg += f"- **Input Dir  :** {args.input}\n"
    finilize_msg += f"- **Output Dir :** {args.output}\n"
    finilize_msg += f"- **Total Items:** {len(processed_items)}\n"
    finilize_msg += f"- **Valid Items:** {valid_cnt}\n"
    finilize_msg = markdown_msg("Finished.", finilize_msg)
    ding_msg(finilize_msg)


if __name__ == "__main__":
    main()

    # EXPORT Object:
    # python .\sproj_helper.py -c obj -e "C:\Program Files\Style3DTestObj2\Style3DTest\Style3DTest.exe" -i "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\sprojs" -o "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\objs-2"
    # python .\sproj_helper.py -c obj -e "C:\Program Files\Style3DTestObj1\Style3DTest\Style3DTest.exe" -i "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\sprojs" -o "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\objs-1"

    # EXPORT SMD:
    # python .\sproj_helper.py -c smd -e "E:\lry\code\style3d_3out\Style3D.exe" -i "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\sprojs" -o "E:\lry\data\style3d_cloud_data\cloud_silhouette\shirt\smds"cache
