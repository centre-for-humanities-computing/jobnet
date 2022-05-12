import os
from pathlib import Path
from typing import List


def get_files(rootdir: str) -> List[str]:

    files_paths = []

    for dir in os.listdir(rootdir):
        dir_path = os.path.join(rootdir, dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            if os.path.isfile(file_path):
                files_paths.append(file_path)
    return files_paths
