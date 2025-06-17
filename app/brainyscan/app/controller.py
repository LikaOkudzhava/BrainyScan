import PIL.Image
import numpy as np
import pydicom.pixel_data_handlers
import pydicom.pixel_data_handlers.util
import pydicom.pixels
import tensorflow as tf
import hashlib
import cv2
import imutils
import PIL
import pydicom
import os
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image

from werkzeug.datastructures import FileStorage

class BrainyController:
    def __init__(self, model_path: str, model_image_size: tuple[int, int]):
        self.model: tf.keras.Model = tf.keras.models.load_model(model_path)
        self.model.trainable = False

        self.image_size = model_image_size
        self.predictions: dict[str, dict[str, float]] = {}
        self.images: dict[str, PIL.Image.Image] = {}

        classes = ('MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented')
        sorted_classes = sorted(classes)

        self.classes: dict[str, int] = {}
        for idx, cl_name in enumerate(sorted_classes):
            self.classes[cl_name] = idx
    
    def __add_prediction(self,
                         id: str,
                         MildDemented: float,
                         ModerateDemented: float,
                         NonDemented: float,
                         VeryMildDemented: float) -> None:
        self.predictions[id] = {
            'MildDemented': MildDemented,
            'ModerateDemented': ModerateDemented,
            'NonDemented': NonDemented,
            'VeryMildDemented': VeryMildDemented
        }
    
    def __get_class_predicted(self,
                              probs: dict[str, float]) -> str:
        return max(probs, key = probs.get)
    
    def __get_sha256_from_filestorage(self, file: FileStorage) -> str:
        """ calculate a checksum of the file object contents binary data

        Args:
            file (FileStorage): file data obect

        Returns:
            str: a checksum value
        """        
        pos = file.stream.tell()
        file.stream.seek(0)

        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.stream.read(4096), b""):
            sha256.update(chunk)
        digest = sha256.hexdigest()

        file.stream.seek(pos)
        return digest
    
    def __get_crop_area(self, image: PIL.Image.Image) -> tuple[int, int, int, int]:
        """Finds the extreme points on the image and returns the bounding
          rectangle for image contents

        Args:
            image (Image.Image): input image data

        Returns:
            _type_: rectangle in format (left, top, right, bottom)
        """        
        input_image = np.array(image)

        gray = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        ADD_PIXELS = 0

        return (
            extLeft[0] - ADD_PIXELS,
            extTop[1] - ADD_PIXELS,
            extRight[0] + ADD_PIXELS,
            extBot[1] + ADD_PIXELS
        )

    def get_stats(self) -> dict[str, int]:
        stats = {
            'MildDemented': 0,
            'ModerateDemented': 0,
            'NonDemented': 0,
            'VeryMildDemented': 0
        }
        for itm in self.predictions.values():
            most_probable = self.__get_class_predicted(itm)
            stats[most_probable] += 1
        
        return stats

    def __jpg_to_image(self, file: FileStorage) -> PIL.Image.Image:
        img = PIL.Image.open(file.stream).convert('RGB')
        return img

    def __dicom_to_image(self, file: FileStorage) -> PIL.Image.Image:
        ds = pydicom.dcmread(file.stream)
        file.stream.seek(0)

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

        img = PIL.Image.fromarray(arr, mode='RGB')
        return img

    def start_predict(self, file: FileStorage) -> dict[str, str] | None:
        id = self.__get_sha256_from_filestorage(file)

        if id not in self.predictions:

            _, extension = os.path.splitext(file.filename)

            img = None
            
            if extension.lower() in ('.jpg', '.jpeg'):
                img = self.__jpg_to_image(file)
            elif extension.lower() in ('.dcm'):
                img = self.__dicom_to_image(file)
            else:
                return None

            self.images[id] = img.copy()
            
            img = img.crop(self.__get_crop_area(img))
            img = img.resize(self.image_size)

            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            probabilities = self.model.predict(x)

            self.__add_prediction(
                id = id,
                MildDemented = probabilities[0][self.classes['MildDemented']].item(),
                ModerateDemented = probabilities[0][self.classes['ModerateDemented']].item(),
                NonDemented = probabilities[0][self.classes['NonDemented']].item(),
                VeryMildDemented = probabilities[0][self.classes['VeryMildDemented']].item()
            )
        return {'id': id }
    
    def get_predict(self, id:str) -> dict[ str, dict[str, float] | str ] | None:
        if id in self.predictions: 
            preds = self.predictions[id]
            return {
                'class': self.__get_class_predicted(preds),
                'probabilities': preds
            }
        return None

    def get_image(self, id: str) -> dict[str, PIL.Image.Image] | None:
        if id in self.images: 
            return {
                'image': self.images[id]
            }
        else:
            print('file not found')
        
        return None
