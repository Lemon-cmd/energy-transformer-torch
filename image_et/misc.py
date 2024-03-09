import os
import re
import glob
from typing import Sequence

EXT = [".pth", ".png", ".pt", ".jpeg", ".jpg", ".txt"]


def make_folder(name: str):
    if not (os.path.exists(name)):
        os.makedirs(name)


def parse_number(name: str) -> int:
    match = re.match(r"([0-9]+).([a-z]+)", name, re.I)
    number = match.groups()[0]
    return int(number)


def get_files(path: str, ext: str = ".pth") -> Sequence[str]:
    if not (os.path.isdir(path)):
        return None

    files = os.listdir(path)

    paths = []
    for basename in files:
        if basename.endswith(ext):
            paths.append(basename)

    if len(paths) == 0:
        return None
    return paths


def get_latest_file(path: str, ext: str = ".pth") -> str:
    files = get_files(path, ext)

    if files is None:
        return files

    return os.path.join(path, max(files, key=parse_number))


def manage_folder(path: str, ext: str, max_num: int = 5):
    files = get_files(path, ext)

    if files is None:
        return

    if len(files) < max_num:
        return

    # delete a quarter number of files
    delete_num = max_num // 4
    delete_num = delete_num if delete_num > 0 else 1

    # oldest to newest files
    files.sort(key=parse_number)

    # remove oldest
    for i in range(delete_num):
        os.remove(os.path.join(path, files[i]))