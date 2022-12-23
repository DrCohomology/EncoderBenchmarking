# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:37:55 2022

@author: federicom
"""
from datetime import date

import glob
import os
import shutil
import re


def pack2server():
    main_version = input("Main version (6 -> pipe tuning, 8 -> no tuning, 9 -> model tuning):")
    lbl = input("Label of experiment:")

    base_dir = r"C:\Users\federicom\Documents\Github\EncoderBenchmarking\src"
    dst_dir = r"C:\Users\federicom\Documents\Github\Experiment_to_server"
    experiment_dir = f"main{main_version}_{date.today().month:02}{date.today().day:02}_{lbl}"

    for dst in [experiment_dir, os.path.join(experiment_dir, "src"), os.path.join(experiment_dir, "results")]:
        try:
            os.mkdir(os.path.join(dst_dir, dst))
        except FileExistsError:
            print(f"{dst} already exists in {dst_dir}!")
            break

    os.chdir(base_dir)

    tocopy = [f"main{main_version}.py", "encoders.py", "utils.py"]
    for base_file in glob.glob(os.path.join("*.py")):
        if base_file not in tocopy:
            continue
        subs = [
            (r"from src\.", "from "),
            (r"import src\.", "import "),
            (r'RESULT_FOLDER = "C:/Data/EncoderBenchmarking_results"', 'RESULT_FOLDER = "./results"')
        ]
        with open(base_file, "r") as fr:
            text = fr.read()
        for local_text, server_text in subs:
            text = re.sub(local_text, server_text, text)

        dst_file = os.path.join(dst_dir, experiment_dir, "src", f"{base_file}")
        with open(dst_file, "w+") as fw:
            fw.write(text)


if __name__ == "__main__":
    pack2server()
