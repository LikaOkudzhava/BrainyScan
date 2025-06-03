#!/usr/bin/env python3

import os, shutil
import zipfile
from zipfile import ZipFile

def zip_add_dir(src_path: str, zip_file: zipfile.ZipFile):
    ''' add dirctory with relative path to the zip archive

    Args:
      src_path: path to the directory
      zip_file: zip archive
    '''
    for root, dirs, files in os.walk(src_path):
        for file in files:
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, src_path)
            zip_file.write(file_path, arcname)

def zip_add_file(
    src_path: str,
    arch_path: str,
    zip_file: zipfile.ZipFile):
    """add file to the zip file archive

    Args:
        src_path (str): file to add
        arch_path (str): file name to be in teh archive
        zip_file (zipfile.ZipFile): zip file object
    """    
    zip_file.write(src_path, arch_path)

def zip_directory(src_dir: str, dst_file: str):
    ''' archive directory

    Args:
      src_dir: path to the directory
      dst_file: name of the archive
    '''
    with zipfile.ZipFile(dst_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zip_add_dir(src_dir, zipf)


def unzip_directory(src_file: str, dst_dir: str) -> str:
    """ unzip archive into a directory

    Args:
        src_file (str): source archive with a full path
        dst_dir (str): destination root dir. data will be 
            upacked into a subdirectory

    Returns:
        str: path to the unpacked data directory
    """    
    filename = os.path.splitext(os.path.basename(src_file))[0]

    dest_path = os.path.join(dst_dir, f'{filename}_extracted')
    shutil.rmtree(dest_path, ignore_errors=True)

    with zipfile.ZipFile(src_file, 'r') as zip_ref:
        zip_ref.extractall(dest_path)
    return dest_path

def append_dir_to_zip(src_dir: str, dst_path: str):
    with zipfile.ZipFile(dst_path, 'a') as zipf:
        zip_add_dir(src_dir, zipf)

def append_file_to_zip(src_path: str, dst_path: str):
    with zipfile.ZipFile(dst_path, 'a') as zipf:
        zip_add_file(src_path, src_path, zipf)

if __name__ == '__main__':
    pass