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
    lbl = input("Label of experiment > ")
    github_path = r"C:\Users\federicom\Documents\Github"
    base_dir = "EncoderBenchmarking"
    SEP_dir = f"EB{date.today().month:02}{date.today().day:02}_{lbl}"

    os.chdir(github_path)

    for dst in [
        SEP_dir,
        os.path.join(SEP_dir, "src"),
        os.path.join(SEP_dir, "results"),
        os.path.join(os.path.join(SEP_dir, "results"), "logs")
    ]:

        try:
            os.mkdir(dst)
        except FileExistsError:
            print(f"{dst} already exists!")
    tocopy = ["main5.py", "encoders.py", "utils.py"]
    os.chdir(os.path.join(base_dir, "src"))
    for fname in glob.glob("*.py"):
        if fname in tocopy:
            with open(fname, "r") as fr:
                x = fr.read()
                y = re.sub(r"from src\.", "from ", x)
                w = re.sub(r"import src\.", "import ", y)
                z = re.sub(
                    r'DATASET_FOLDER = "C:/Data"', 'DATASET_FOLDER = "./../data"', w
                )
                z2 = re.sub(
                    r"os.chdir(\'C:/Users/federicom/Documents/Github/EncoderBenchmarking\')", '', z  
                )
                z3 = re.sub(
                    r'RESULT_FOLDER = "C:/Data/EncoderBenchmarking_results"', 'RESULT_FOLDER = "./results"', z2
                )
            filedst = os.path.join(os.path.join(github_path, SEP_dir, "src"), fname)
            with open(filedst, "w+") as fw:
                fw.write(z3)


if __name__ == "__main__":
    pack2server()
    print(
        "Before running the experiment, double check: utils.tune_pipe, utils.DATASET_FOLDER and main1.encoders_list"
    )
