

\# WildlifeAI 🐾



AI-powered wildlife detection and species classification using Python and deep learning.



\## About the Project



WildlifeAI analyzes wildlife images, detects animals, and predicts their species. The project uses MegaDetector to locate animals and a ResNet18 image classifier to predict the species.



The goal is to build a camera-trap system that can help identify Indian wildlife automatically.



\## Features



\- Animal detection using MegaDetector V6

\- Species classification using ResNet18

\- GPU acceleration with NVIDIA CUDA when available

\- Automatic processing of images in the images folder

\- JSON reports containing predictions and confidence scores

\- Dataset checking and model training scripts



\## Technologies Used



\- Python

\- PyTorch and TorchVision

\- PytorchWildlife

\- Pillow

\- MegaDetector V6

\- ResNet18



\## Project Structure



```text

WildlifeAI/

├── dataset/

│   ├── train/

│   └── val/

├── images/

├── models/

├── results/

├── check\_dataset.py

├── classify\_test.py

├── detect.py

├── predict.py

├── train.py

├── wildlife\_pipeline.py

├── requirements.txt

└── README.md

```



Note: Dataset images, trained models, input images, and generated results are kept locally and are excluded from Git.



\## Setup



1\. Install Python 3.12.

2\. Clone this repository:



&#x20;  ```bash

&#x20;  git clone https://github.com/Lucid-v/WildlifeAI.git

&#x20;  cd WildlifeAI

&#x20;  ```



3\. Create and activate a virtual environment on Windows:



&#x20;  ```powershell

&#x20;  py -3.12 -m venv .venv

&#x20;  .\\.venv\\Scripts\\Activate.ps1

&#x20;  ```



4\. Install dependencies:



&#x20;  ```bash

&#x20;  python -m pip install --upgrade pip

&#x20;  pip install -r requirements.txt

&#x20;  ```



\## Usage



Place wildlife images inside the `images/` folder.



Run animal detection:



```bash

python detect.py

```



Train the species classifier:



```bash

python train.py

```



Predict the species in a test image:



```bash

python predict.py

```



Run the complete detection and classification pipeline:



```bash

python wildlife\_pipeline.py

```



The training and prediction scripts require the appropriate dataset and trained model files to be present locally.



\## Current Status



This project is an experimental prototype. Species predictions depend on the quality, size, and variety of the training dataset. Confidence scores should not be treated as verified animal identifications.



\## Future Improvements



\- Add a larger, accurately labelled Indian wildlife dataset

\- Evaluate classification accuracy

\- Build a FastAPI backend for image uploads

\- Integrate an ESP32-CAM

\- Create a web dashboard for wildlife monitoring



\## Author



Varad



\## License



No license has been added yet. Contact the author before reusing or distributing this project.



