import bentoml
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.adapters import ImageInput
import numpy as np
from keras.preprocessing import image
from PIL import Image


# @env(pip_packages=['fastai'])
# @artifacts([Fastai1ModelArtifact('pet_classifer')])
# class PetClassifier(BentoService):
    
#     @api(input=ImageInput(), batch=False)
#     def predict(self, image):
#         fastai_image = pil2tensor(image, np.float32)
#         fastai_image = Image(fastai_image)
#         result = self.artifacts.pet_classifer.predict(image)
#         return str(result)
CLASS_NAMES = ['cat', 'dog']

@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([KerasModelArtifact('model')])
class PetClassifier(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self,  imageInput):
        img = Image.fromarray(imageInput)
        img= img.resize((150, 150)) 
        img_tensor = np.array(img, dtype = float)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.
        prediction = self.artifacts.model.predict(img_tensor)
        print("prediction2\n")
        print(prediction)
        return prediction


#  inputs = []
#         for img in imgs:
#             img = Image.fromarray(img).resize((28, 28))
#             img = np.array(img.getdata()).reshape((28, 28, 1))
#             inputs.append(img)
#         inputs = np.stack(inputs)
#         class_idxs = self.artifacts.classifier.predict_classes(inputs)
#         return [class_names[class_idx] for class_idx in class_idxs]