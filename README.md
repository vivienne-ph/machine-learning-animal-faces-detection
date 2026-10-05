# machine-learning-animal-faces-detection
A machine learning system that decides whether to open an outdoor pet feeder's tray, based on whether the animal in front of its camera is a target pet (cat/dog) or a non-target wild animal.

## Overview
- Task	3-class image classification: cat, dog, wild - used to drive a binary feed/no-feed decision
- Dataset	AFHQ (Animal Faces-HQ), NVIDIA / StarGAN v2 (CVPR 2020) - ~15,800 images, balanced across 3 classes
- Model	MobileNetV2 pretrained on ImageNet, backbone frozen, only the final classifier layer trained
- Why MobileNetV2	Designed for embedded/IoT deployment - matches a feeder device mounted outdoors, not a server
- Primary metric	Macro F2-score (weights Recall over Precision - missing a real pet is worse than feeding a wild animal)

## Theoretical Framework

The project is explicitly mapped onto the 4 pillars of machine learning:

| Pillar | Role | This project |
|---|---|---|
| **S - Data** | Observed examples (X, Y) | AFHQ, stratified 70/15/15 split |
| **H - Hypothesis Space** | Candidate functions the model can represent | Frozen MobileNetV2 backbone → linear classifier (H restricted to ~3.8K free parameters) |
| **L - Criterion** | Loss function | CrossEntropyLoss (standard) vs. class-weighted CrossEntropyLoss (cost-sensitive variant, tested) |
| **A - Algorithm** | Optimization procedure | Adam, applied only to the classifier's parameters |

## Key Design Decision: Cost-Sensitive Evaluation

- Standard accuracy rewards degenerate policies here — e.g. a model that always predicts "cat" still scores well on naive accuracy while never correctly recognizing a dog. The project instead:
- Defines an explicit expected cost metric, assuming a false negative (missing a real pet) costs more than a false positive (misidentifying a wild animal) - cost ratio tested at 2:1, 5:1, and 10:1 for sensitivity.
- Compares the trained model against degenerate baselines (always-open, always-closed, random) to prove it isn't just exploiting class imbalance.
- Tunes a decision threshold (not just argmax) on the validation set only, to reduce missed-pet errors without touching the test set.
- Reports Clopper-Pearson confidence intervals on small real-world test samples, instead of treating a single run's accuracy as ground truth.
- Trains 5 random seeds × 4 configurations (CrossEntropy / class-weighted / ResNet18 baseline / etc.) to confirm results aren't a lucky single run.

## Results (test set, n = 2,371)
| Metric | Value |
|---|---|
| Macro F2-score (3-class) | 0.9949 |
| Accuracy | 0.9949 |
| Pet (cat/dog) Recall — 95% CI | 0.9975 (0.9937, 0.9993) |
| False negatives (pet → wild) | 4 |
| False positives (wild → pet) | 7 |
| Cat ↔ Dog confusions | 1 (Cat → Dog only; 0 Dog → Cat) |

**Degenerate-baseline sanity check** - confirms the model isn't just exploiting class imbalance:

| Policy | Accuracy | Pet Recall | Expected Cost |
|---|---|---|---|
| **Trained model** | **0.9949** | **0.9975** | **27** |
| Always predict majority class (Cat) | 0.3518 | 1.0000 | 761 |
| Uniform random | 0.3273 | 0.6733 | 3,161 |
| Always predict Wild | 0.3210 | 0.0000 | 8,050 |

<img width="668" height="215" alt="image" src="https://github.com/user-attachments/assets/6b19fec3-0651-4d82-a66c-92d9cd3fde85" />
<img width="302" height="250" alt="image" src="https://github.com/user-attachments/assets/a0de3733-3fdf-4871-9b71-dd7bb8efd2d3" />
<img width="353" height="218" alt="image" src="https://github.com/user-attachments/assets/cecc7b6d-94ec-458e-9c3a-61b2b1c50b5d" />

## Model Interpretability (Grad-CAM)
 <img width="472" height="290" alt="image" src="https://github.com/user-attachments/assets/4575cbb0-7500-45ee-a243-c15fd9562bf2" />

Grad-CAM was applied to the final convolutional layer to verify the model attends to biologically meaningful facial features (eyes, nose, muzzle) rather than background artifacts, checked across 15 AFHQ samples and cross-validated against informally captured real-world photos.

## Real-World Validation
<img width="466" height="364" alt="image" src="https://github.com/user-attachments/assets/8223ce3c-efc4-4df7-a0d8-0ebcccca62a6" />

Beyond the AFHQ test set, the trained model was tested on personal photos taken outside the training distribution - different angles, clothing, multiple animals in frame, and genuinely unfamiliar animals (raccoon, quokka, wombat) to probe generalization to the "wild" class. Key finding: confidence drops noticeably when the animal's face isn't clearly visible, a real limitation for outdoor deployment, since AFHQ consists entirely of close-up, front-facing shots.

## Known Limitations
AFHQ is a clean, curated dataset: real feeder camera footage will have wider angles, variable outdoor lighting, and partial occlusion.
The model is a whole-image classifier, not an object detector, it has no mechanism to handle frames containing multiple animals of different classes.
Real-world validation is small-scale and informal (not a systematic benchmark).
The 5:1 cost ratio used for the decision threshold is an assumption, not a measured business cost.

 ## Running the Project

The notebook is fully self-contained - it downloads the dataset and installs dependencies automatically.

Runtime → Change runtime type → GPU (T4)
Run all cells top to bottom. The dataset downloads automatically via kagglehub.

📁 Repo Structure

├── animal-faces-detection.ipynb       # Full notebook: data, model, training, evaluation, Grad-CAM

└── README.md


AFHQ dataset: Choi, Y., Uh, Y., Yoo, J., & Ha, J.-W. (2020). StarGAN v2: Diverse Image Synthesis for Multiple Domains. CVPR. Dataset mirror used: dimensi0n/afhq-512 on Kaggle.
