# Trustworthiness and Uncertainty in Medical AI: \ A Case Study Using a Diabetic Retinopathy Dataset

![Project Thumbnail](figures/01Thumbnail.png)

Minkyeong Kim<sup>†</sup>, Minseok Han<sup>†</sup>, Hojun Jeong<sup>†</sup>, Hyunmin Choi<sup>†</sup>, Junsoo Seo<sup>†</sup>, Hyungjin Yoon<sup>†</sup>  

<sub><sup>†</sup> Equal contribution, alphabetically ordered (Korean)</sub>

 
---

### Summary

![workflow](figures/03Workflow.png)

This project builds a reliable end-to-end pipeline for diabetic retinopathy (DR) screening that centers data quality, calibration, and interpretability. We re-labeled RetinaMNIST using ICDR grades (0–4), collapsing grades 1–4 into DR-positive and grade 0 into non-DR to form a clinically meaningful binary task. To control input quality, we targeted common optical artifacts: we first detected the circular fundus boundary and blackened all pixels outside it to remove text/markings; then we screened for crescent-shaped flare by comparing mean intensities in the outer ring (top/bottom/left/right) against the center. These procedures (see p.11 and the logic in `src/Preprocess.py`) reduce non-pathological cues and lighting bias before training.

![dat preprocessing](figures/02Data_preprocessing.png)

We trained ResNet-50 on RetinaMNIST and then transferred/fine-tuned to the Brazil mBRSET dataset to assess domain robustness (portable camera vs. curated data). Generalization was strengthened with a diverse augmentation stack (as on p.12): rotations, horizontal/vertical flips, random crop/resize, color jitter, and random erasing. Qualitatively, Grad-CAM maps show that augmentation helps the model focus on clinically plausible retinal regions across DR-positive and control cases.

![augmentations](figures/12Types_of_Augmentations_used.png)

Beyond accuracy, we emphasized trustworthy probabilities. We implemented deep ensembles and Monte Carlo Dropout (p ∈ {0.001, 0.01, 0.1, 0.2}) and evaluated Expected Calibration Error (ECE), Brier score, and Negative Log-Likelihood (NLL). MC Dropout yielded only marginal improvements in Brier and ECE—alongside small gains in Accuracy within our uncertainty-evaluation setup—while we also observed a slight decrease in peak accuracy compared with a fully committed CNN trained/evaluated without dropout. We therefore explored multiple dropout rates and ensemble sizes to characterize this trade-off. Grad-CAM visualizations complement these findings by highlighting decision-relevant retinal regions, supporting transparent model behavior.

![ensemble outline](figures/06Ensemble_outline.png)
![mc dropout outline](figures/07MC_dropout_outline.png)

---

### Overview & Contributions

* End-to-end DR classification pipeline with transfer learning.
* Careful preprocessing for fundus boundary detection and artifact removal.
* Data augmentation strategies to improve generalization.
* Uncertainty quantification through deep ensembles and MC Dropout.
* Calibration assessment using ECE, Brier score, and NLL.
* Grad-CAM visualizations for interpretability.

---

### Repository Structure

```
notebooks/   # training, evaluation, and figure reproduction
src/         # models, dataset loaders, preprocessing, training loops
utils/       # metrics, plotting, helper functions
```

---

### Data Setup

* **RetinaMNIST**: downloaded and re-labeled using ICDR 0–4 scale (0 = non-DR, 1–4 = DR).
* **mBRSET**: portable-camera DR dataset for external validation.
* Preprocessing (p.11): fundus outline detection, blackening outside pixels, crescent artifact removal.

---

### Training Procedure

* **Stage 1 (RetinaMNIST)**: Train ResNet-50 with/without augmentation.
* **Stage 2 (mBRSET)**: Transfer learning and fine-tuning.
* **Augmentations (p.12)**: rotation, flipping, cropping/resizing, color jitter, random erasing.

---

### Uncertainty & Calibration

* **Deep Ensembles**: train multiple ResNet-50 instances.
* **MC Dropout**: inference with dropout p = {0.001, 0.01, 0.1, 0.2}.
* Metrics: Expected Calibration Error (ECE), Brier score, Negative Log-Likelihood (NLL).
* Trade-off: marginal metric gains but slight decrease in peak accuracy.

---

### Results

* **Accuracy and F1**: improved with augmentation and transfer learning.
![reuslts1](figures/04No_Aug_vs_Aug.png)
![reuslts2](figures/05No_Aug_vs_Aug_2.png)
* **Uncertainty metrics**: MC Dropout gave small improvements in Brier/ECE, ensembles more stable.
![reuslts3](figures/09CNN_ensemble_metrics.png)
![reuslts4](figures/10CNN_ensemble_uncertainty_metric.png)
![results5](figures/11MC_dropout_uncertainty_metric.png)
* **Limitations**: minor decrease in accuracy compared to dropout-free CNN baselines.

---

### Citation

If you use this repository, please cite as:

```bibtex
@misc{digitalhealthcarebootcamp2025,
  title        = {Trustworthiness and Uncertainty in Medical AI: A Case Study Using a Diabetic Retinopathy Dataset},
  author       = {Kim, Minkyeong and Han, Minseok and Jeong, Hojun and Choi, Hyunmin and Seo, Junsoo and Yoon, Hyungjin},
  year         = {2025},
  note         = {Digital Health Care Bootcamp 2025, Yonsei University College of Medicine},
  url          = {https://github.com/youravgENTP/DigitalHeatlhCareBootcamp_2025_Winners}
}
```

---

### References

- Wong TY, Cheung CMG, Larsen M, Sharma S, Simó R. *Diabetic retinopathy.* Nat Rev Dis Primers. 2016;2:16012.  
- Yang J, Shi R, Wei D, Liu Z, Zhao L, Ke B, et al. *MedMNIST v2 – A large-scale lightweight benchmark for 2D and 3D biomedical image classification.* Sci Data. 2023;10:41.  
- Liu R, Wang X, Wu Q, Dai L, Fang X, Yan T, et al. *DeepDRiD: Diabetic Retinopathy—Grading and Image Quality Estimation Challenge.* Patterns. 2022;3(6):100512.  
- Lakshminarayanan B, Pritzel A, Blundell C. *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* arXiv [preprint]. 2016 Dec 5;abs/1612.01474.  
- Jospin LV, Buntine W, Boussaid F, Laga H, Bennamoun M. *Hands-on Bayesian Neural Networks – a Tutorial for Deep Learning Users.* arXiv [preprint]. 2020 Jul 14 [cited 2025 Jul 24];v1.  
- Curran K, Peto T, Jonas JB, Friedman D, Kim JE, Leasher J, Tapply I, Fernandes AG, Cicinelli MV, Arrigo A, Leveziel N, Resnikoff S, Taylor HR, Sedighi T, Flaxman S, Bikbov MM, Braithwaite T, Bron A, Cheng C, … Zheng P. (2024). *Global estimates on the number of people blind or visually impaired by diabetic retinopathy: a meta-analysis from 2000 to 2020.* Eye, 38(11), 2047–2057. https://doi.org/10.1038/s41433-024-03101-5  

