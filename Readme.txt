Zero-Shot Anomaly Detection using Vision Foundation Models

This project explores how recent vision foundation models such as DINOv3, the mirroring DINO model and/or SAM can be applied for zero-shot and few-shot anomaly detection on industrial data. Students will use pre-trained models to detect surface defects and irregular textures in the MVTec Anomaly Detection (AD) dataset without additional training — by comparing patch-level embeddings between normal and anomalous samples.


Prerequisits to run the code:
 - open terminal in the base folder "ZeroShot_Anomaly..."
 - "pip install git+https://github.com/facebookresearch/segment-anything.git"
 - run "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
 - "pip install opencv-python pycocotools matplotlib onnxruntime onnx"

 - Download all the required packages or run the command below to create a Conda environment
  - "conda env create -f environment.yml"

Main Jupyter notebook to reproduce our results:
 - ZeroShot_v2.0.ipynb



Data: MVTec AD dataset (publicly available; already preprocessed for easy use).
Download the dataset and put it in the main folder, such that the structure is
/ZeroShot_Anomaly_Detection_using_Vision_Foundation_Models/
mvtec_anomaly_detection/

Models:
https://github.com/facebookresearch/dinov3
https://github.com/facebookresearch/segment-anything

Dataset:
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads

Code inspiration:
https://github.com/Kaisor-Yuan/AD-DINOv3
