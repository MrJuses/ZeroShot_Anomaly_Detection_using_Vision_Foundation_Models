    Zero-Shot Anomaly Detection using Vision Foundation Models

This project explores how recent vision foundation models such as DINOv3, the mirroring DINO model and/or SAM can be applied for zero-shot and few-shot anomaly detection on industrial data. Students will use pre-trained models to detect surface defects and irregular textures in the MVTec Anomaly Detection (AD) dataset without additional training — by comparing patch-level embeddings between normal and anomalous samples.


Optional extensions include testing fine-tuning or prompt-based feature adaptation for improved results.

Key Tasks:

    Use DINOv3 to extract embeddings or mirroring DINO and SAM for extra help.
    Compare normal vs. defective samples using similarity metrics.
    Visualize and interpret anomaly maps.
    Evaluate performance on the MVTec AD dataset.

 

Students will learn to:

    Apply large pre-trained vision models for industrial inspection.
    Use zero-shot and few-shot learning strategies.
    Perform embedding-based similarity and visualization in PyTorch.
    Evaluate unsupervised anomaly detection results.


Data: MVTec AD dataset (publicly available; already preprocessed for easy use).

model:
https://github.com/facebookresearch/dinov3
https://github.com/facebookresearch/segment-anything

Code:
https://github.com/Kaisor-Yuan/AD-DINOv3

dataset:
https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads


Prerequisits to run the code:
 - open terminal in the base folder "ZeroShot_Anomaly..."
 - "pip install git+https://github.com/facebookresearch/segment-anything.git"
 - run "wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
 - "pip install opencv-python pycocotools matplotlib onnxruntime onnx"

Main code:
 - ZeroShot_v2.0.ipynb
 
