# Cat Breed Classifier

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-ML-orange.svg)
![Status](https://img.shields.io/badge/Project-Completed-green.svg)

---

## Overview
Deep learning project to classify cat breeds from images using **MobileNetV2 (Transfer Learning)**.

- 12 cat breed classification
- Top-3 predictions with confidence scores
- Streamlit-based simple UI

---

## Model
- MobileNetV2 (pretrained)
- Custom dense layers added
- Output: 12 cat breeds
- Trained on Oxford-IIIT Pet dataset (cat subset)

---

## Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Streamlit

---

## Workflow
Image → Preprocessing → Model Prediction → Top-3 Results → Breed Output

---

## Output Example
![1](/output/1.png)
![2](/output/2.png)

---

## Features
- Image upload prediction
- Top-3 breed results
- Confidence scores

---

## Future Work
- Webcam prediction
- Model improvement
- Cloud deployment

## Oxford-IIIT Pet Dataset Used
Link: https://www.robots.ox.ac.uk/~vgg/data/pets/

### Cat Breeds (12 classes)
Abyssinian
Bengal
Birman
Bombay
British Shorthair
Egyptian Mau
Maine Coon
Persian
Ragdoll
Russian Blue
Siamese
Sphynx

### Dog Breeds (25 classes) - NOT USED
American Bulldog
American Pit Bull Terrier
Basset Hound
Beagle
Boxer
Chihuahua
English Cocker Spaniel
English Setter
German Shorthaired
Great Pyrenees
Havanese
Japanese Chin
Keeshond
Leonberger
Miniature Pinscher
Newfoundland
Pomeranian
Pug
Saint Bernard
Samoyed
Scottish Terrier
Shiba Inu
Staffordshire Bull Terrier
Wheaten Terrier
Yorkshire Terrier

