# BraTS_2021

Our implementation of the BraTS 2021 challenge.

```
The Brain Tumor Segmentation (BraTS) challenge celebrates its 10th anniversary, and this year is jointly organized by the Radiological Society of North America (RSNA), the American Society of Neuroradiology (ASNR), and the Medical Image Computing and Computer Assisted Interventions (MICCAI) society.

The RSNA-ASNR-MICCAI BraTS 2021 challenge utilizes multi-institutional pre-operative baseline multi-parametric magnetic resonance imaging (mpMRI) scans, and focuses on the evaluation of state-of-the-art methods for (Task 1) the segmentation of intrinsically heterogeneous brain glioblastoma sub-regions in mpMRI scans. Furthemore, this BraTS 2021 challenge also focuses on the evaluation of (Task 2) classification methods to predict the MGMT promoter methylation status.
```

## To Do List 

### 27 July

* Try to understand the problem, is it a binary classification problem or image segmentation ? **Done**
* Try to understand the input **Done**
* One model per tool -> then maybe agregate the results ?
* One model per tool -> one submission per tool
* Dataloader in pytorch to load dcm images / normalize input (256 pixels) **Done**
* Learnable data normalization

### 28 July

#### Approaches that we can try:

* Baseline : Resnet (binary Classification) using one tool

### 29 July

* RNN with vec size 500



#### Input files

