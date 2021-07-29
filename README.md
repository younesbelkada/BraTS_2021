# BraTS_2021

Our implementation of the BraTS 2021 challenge.

```
The Brain Tumor Segmentation (BraTS) challenge celebrates its 10th anniversary, and this year is jointly organized by the Radiological Society of North America (RSNA), the American Society of Neuroradiology (ASNR), and the Medical Image Computing and Computer Assisted Interventions (MICCAI) society.

The RSNA-ASNR-MICCAI BraTS 2021 challenge utilizes multi-institutional pre-operative baseline multi-parametric magnetic resonance imaging (mpMRI) scans, and focuses on the evaluation of state-of-the-art methods for (Task 1) the segmentation of intrinsically heterogeneous brain glioblastoma sub-regions in mpMRI scans. Furthemore, this BraTS 2021 challenge also focuses on the evaluation of (Task 2) classification methods to predict the MGMT promoter methylation status.
```

## Architecture :
```
├── data                            # BraTS dataset from kaggle
│   ├── train                       # 586 samples
│       ├── FLAIR                   # describe FLAIR
│           ├── yyyyyy              # index of individual 
│               ├── Image-xxx.dcm   # dcm images
│       ├── T1w                     # describe
│       ├── T1wCE                   # describe
│       ├── T2w                     # describe
│   ├── test                        # only available on Arthur"s computer
│   ├── train_labels.csv            # only available on Arthur"s computer
│   ├── sample_submission.csv       # only available on Arthur"s computer
├── dataset						    # Files to process data, dataset loaders
├── experiments 				    # checkpoints and results of trainig
│   ├── name_of_experiment
├── agents                          # network agents
│   ├── baseline.py                     # defines model,dataloader, loss, optimizer...
├── config 
│   ├── name.yaml                   # parameters for runx experiment
├── graphs 
│   ├── models                      # actual model codes used in agents
│       ├── baseline.py
│   ├── loss                        # loss functions and their implementation
├── pretrained_weights
├── utils                           # utility files for measure, runx etc
├── main.py                         # main used to launch 
├── run.sh                          # run experiment (using runx)

```
### The agent
The agent controls the training and evaluation process of your model and is considered the core of the project.


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

* RNN with vec size 500 -> arthur will do this man
* Understand how to deal with video data with arbitrary variable number of frames


#### Input files

