> This repository contains the code for **ECE 395 & 396: Senior Electrical Engineering Projects**, a two-semester course offered in Fall 2024 and Spring 2025 at The Cooper Union for the Advancement of Arts and Science, featuring our project exploring the impact of quantization on face recognition models to enhance performance and explainability on low-powered devices.


## Senior Projects
**Course, Fall 2024 & Spring 2025**  
**Instructor:** Professor Sam Keene   


###  Overview

This course is a year-long, hands-on journey where students work in small groups to tackle real-world challenges in electrical and computer engineering. From project planning and budgeting to designing systems, building hardware and software, and evaluating performance, students cover it all. Along the way, they also sharpen their communication skills through regular presentations and written reports, preparing them for future research or industry roles.


### Repository Structure
```
.
├── Final_Project
│   ├── README.md
│   ├── app.py
│   ├── hubconf.py
│   ├── main.py
│   ├── requirements.txt
│   ├── run.sh
│   └── setup.sh
├── README.md
└── artifacts
```

### Final Project
**Team Members:** Diego Toribio & Nicholas Storniolo

The project, *Low-Powered Face Misclassification: Toward Ethical & Efficient On-Device Facial Recognition*, investigates how post-training and quantization-aware techniques reshape face-ID models for edge hardware. We first fine-tune ResNet-18 and Vision Transformer backbones on the VGGFace2 dataset to establish FP32 baselines for identification and verification accuracy. Next, we sweep FP16 and INT8 variants—leveraging PyTorch + TensorRT—to measure speed-ups, energy draw, and memory savings on a Jetson Nano. Throughout the sweep we generate saliency maps (Grad-CAM for CNNs, attention rollout for ViTs) to see whether quantization alters which facial regions drive decisions and where misclassifications arise. The end deliverable is a compact benchmark suite and set of guidelines that balance privacy, interpretability, and performance for real-time, on-device facial recognition.
