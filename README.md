# Pet Classifier CNN and served with BentoML

In this small project, I build a small classifier where based on the picture, it will determine if it's a dog or cat. The accuracy does not seem to be the best, but as I was mostly interested in practicing the concepts and using BentoML, that is not very important to me now. However in the near future I plan to train some other models and pick the one with the best accuracy.

## Setup:
In the root of the project please create this structure:
```java
PetImages 
  |--->Cat
  |--->Dog
```

and put in there the dataset.

### Create test, train and validation data sets
Run `python create_test_train_dir.py` to create the data sets.

### Build the model and attach it to a BentoML service:
Run `python pet_classifier_model.py` to build the CNN model and attach it to BentoML service.

### Serve the service:
Run `bentoml serve PetClassifier:latest` to serve the service locally on port 5000 (Go to `127.0.0.1:5000` to see the swagger documentation and try it out.)

## Credits:
1. https://github.com/abaranovskis-redsamurai/automation-repo/tree/master/convnet - source I used for the model
2. `find_broken_images.py` was found somewhere in the internet, and I no longer can find the real source. Kudos to the person who wrote that piece of code.