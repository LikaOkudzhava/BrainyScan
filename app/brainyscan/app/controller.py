from PIL import Image
import numpy as np
import tensorflow as tf
import hashlib
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.preprocessing import image

from werkzeug.datastructures import FileStorage

class BrainyController:
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)

        self.image_size = (299, 299)
        self.predictions = {}

        classes = ('MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented')
        sorted_classes = sorted(classes)

        self.classes = {}
        for idx, cl_name in enumerate(sorted_classes):
            self.classes[cl_name] = idx
    
    def __add_prediction(self,
                         id: str,
                         MildDemented: float,
                         ModerateDemented: float,
                         NonDemented: float,
                         VeryMildDemented: float):
        self.predictions[id] = {
            'MildDemented': MildDemented,
            'ModerateDemented': ModerateDemented,
            'NonDemented': NonDemented,
            'VeryMildDemented': VeryMildDemented
        }
    
    def __get_class_predicted(self,
                              probs: dict[str, float]) -> str:
        return max(probs, key = probs.get)
    
    def __get_sha256_from_filestorage(self, file: FileStorage) -> int:
        pos = file.stream.tell()
        file.stream.seek(0)

        sha256 = hashlib.sha256()
        for chunk in iter(lambda: file.stream.read(4096), b""):
            sha256.update(chunk)
        digest = sha256.hexdigest()

        file.stream.seek(pos)
        return digest

    def get_stats(self):
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

    def start_predict(self, file: FileStorage) -> dict:
        if file is None:
            return {"error": "No file provided"}, 400

        id = self.__get_sha256_from_filestorage(file)

        if id not in self.predictions:
            img = Image.open(file.stream).convert('RGB')
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
    
    def get_predict(self, id: str):
        if id in self.predictions: 
            preds = self.predictions[id]
            return {
                'class': self.__get_class_predicted(preds),
                'probabilities': preds
            }
        
        return None
