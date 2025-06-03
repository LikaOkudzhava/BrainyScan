#!/usr/bin/env python3

import os

def is_colab() -> bool:
    """checks, is current working environment a google collab

    Returns:
        bool: True, if script executed by google collab
    """    
    result = False
    try:
        import google.colab
        result = True
    except ImportError:
        result = False
    return result

def mount_gdrive(mount_point: str) -> str:
    """mount gdrive

    Args:
        mount_point (str): mount point for gdrive

    Returns:
        str: path to the gdrive contents
    """    
    from google.colab import drive

    drive.mount(mount_point, )
    return os.path.join(mount_point, 'MyDrive')
