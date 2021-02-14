import bentoml
from bentoml.frameworks.keras import KerasModelArtifact
from bentoml.adapters import ImageInput
import numpy as np
from keras.preprocessing import image
from PIL import Image


@bentoml.env(infer_pip_packages=True)
@bentoml.artifacts([KerasModelArtifact("model")])
class PetClassifier(bentoml.BentoService):
    @bentoml.api(input=ImageInput(), batch=False)
    def predict(self, imageInput):
        img = Image.fromarray(imageInput)
        img = img.resize((150, 150))
        img_tensor = np.array(img, dtype=float)

        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0
        prediction = self.artifacts.model.predict(img_tensor)
        value = prediction[0][0]
        if value > 0.5:
            return "Dog"
        else:
            return "Cat"
