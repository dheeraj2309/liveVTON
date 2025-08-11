# 👗 Virtual Try-On (VTON) & Clothing Recommendation Model

This repository contains the implementation of a **Virtual Try-On (VTON)** system combined with a **clothing recommendation model**. It provides an interactive interface to try on clothes virtually and receive recommendations based on clothing attributes.

# Virtual Try-On
## 🔗 Checkpoints and 📦 Dataset
Pre-trained model checkpoints are available **[here](<https://drive.google.com/file/d/1Fnq10Rnn5Y_ns_StAZWyDp2gQcqSRqrU/view?usp=sharing>)**.

After downloading, place them in: checkpoints folder in app folder

## Super-Resolution
The super resolution model can be donwloaded from **[this link](<https://drive.google.com/file/d/1BJS9t1MrNtogHVVsiFSU5gxPCNlnH7cW/view?usp=sharing>)**.

After downloading, unzip the folder and place the super-resolution model in VTON-Model folder.

## ⚙️ Installation
pip install -r requirements.txt
Also in the Real_ESRGAN do "pip install -r requirements.txt"

## ⚙️ Run
run the flutter app: python main_app.py for app folder.

# Recommedation system
## 📦 Dataset
The dataset required for this project can be downloaded from **[this link](<https://drive.google.com/file/d/1aG44QCPNlVLnD61tJPCpfP8fWE9k3Til/view?usp=sharing>)**.
After downloading,  make a pants_dataset folder and unzip the folder there. 

## 📦 .env
rename .env.example in FASHIONRECOMMENDER to .env and add your own gemini api key.

## ⚙️ Run

Paste all the files from FASHIONRECOMMENDER into VTON-model.
run the flask app: python flask_ap.py for Recommendation system
 






