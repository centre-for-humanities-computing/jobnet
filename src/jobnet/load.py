import os
from os import listdir
from os.path import isfile, join
import pathlib
from pathlib import Path


def get_files(path: str) -> str:

    files = []

    for item in listdir(path):
        full_path = os.path.join(path, item)
        if os.path.isfile(full_path):
            files.append(full_path)
    return files
