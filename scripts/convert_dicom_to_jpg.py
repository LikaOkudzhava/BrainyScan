#!/usr/bin/env python3

import os
import pydicom
import pydicom.pixel_data_handlers
import pydicom.pixel_data_handlers.util
import pydicom.multival
import numpy as np
from PIL import Image

class Convert:
    
    def convert(self, src_dcm: str, dst_jpg: str):
        """convert image from DICOM format to the RGB JPG

        Args:
            src_dcm (str): source DCM image path
            dst_jpg (str): destination JPG image path
        """
        ds = pydicom.dcmread(src_dcm)

        arr = pydicom.pixel_data_handlers.util.apply_voi_lut(
                ds.pixel_array, ds
            ) if hasattr(ds, 'VOILUTSequence') else ds.pixel_array

        if ds.PhotometricInterpretation.startswith("MONOCHROME") or len(arr.shape) == 2:
            arr = np.stack([arr]*3, axis=-1)

        elif ds.PhotometricInterpretation in ['YBR_FULL', 'YBR_FULL_422']: # if non-rgb space
            arr = pydicom.pixel_data_handlers.convert_color_space(
                arr, ds.PhotometricInterpretation, 'RGB')
        
        if arr.dtype != np.uint8:   # if it is not 0-255, normalize
            arr = arr.astype(np.float32)
            arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
            arr = arr.astype(np.uint8)

        img = Image.fromarray(arr, mode='RGB')
        img.save(dst_jpg)

    def start(self, src_dir: str, dst_dir: str, add_dirs: bool = False):
        """convert all the dcm file in the source directory 
            to the jpg in destination directory keeping the
            source directory structure.

        Args:
            src_dir (str): source directory
            dst_dir (str): destination directory
            add_dirs (bool): if true, the parent directories 
                names will be added to a jpg file names
        """        
        for path, _, files in os.walk(src_dir):
            rel_path = os.path.relpath(path, src_dir)
            dst_path = os.path.join(dst_dir, rel_path)
            os.makedirs(dst_path, exist_ok=True)
            for f in files:
                fname, ext = os.path.splitext(f)
                if ext.lower() == '.dcm':
                    if add_dirs:
                        fname = f'{rel_path.replace(os.sep, "_")}_{fname}'

                    self.convert(
                        os.path.join(path, f),
                        os.path.join(dst_path, f'{fname.lower()}.jpg')
                    )
        return
            
if __name__ == '__main__':
    c = Convert()
    c.start("data/AlzheimersDICOM", "data/AlzheimersDICOM_jpeg", True)