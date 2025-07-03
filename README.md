# HECKTOR2025 - Challenge

<p align="center">
  <img src="HECKTOR-main.jpeg">
</p>

Welcome to the **HECKTOR 2025 Challenge** repository! This repository contains instructions and examples for creating a baseline and a valid docker for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). It will also help you with how you can submit your designed model to the [Grand Challenge](https://hecktor25.grand-challenge.org/hecktor25/) for evaluation. Here you’ll find everything you need to get started quickly: from understanding the challenge, to setting up your environment, training your first model, and evaluating your results. So this reporsitory has **two primary branches** 🌲:

- [**main**](https://github.com/BioMedIA-MBZUAI/HECKTOR2025/tree/main): Here you’ll find step-by-step guides, data loaders, training scripts, and inference examples so you can get a working model up and running in minutes.

- [**docker-template**](https://github.com/BioMedIA-MBZUAI/HECKTOR2025/tree/docker-template): Designed for containerizing and submitting your final models to the Grand Challenge. This branch provides a Docker-based inference template, build/test/save scripts, and enforces all challenge restrictions.
---

# How can this Repo help?

1. Understand what the challenge is about  
2. Set up your development environment  
3. Train models on our provided data  
4. Test and evaluate your results  
5. Explore ideas for improving performance 

---
# 🚀 About the HECKTOR'25 Challenge
Head and Neck (H&N) cancers are among the most common cancers worldwide (5th leading cancer by incidence) [Parkin et al. 2005]. Radiotherapy combined with cetuximab has been established as a standard treatment [Bonner et al. 2010]. However, locoregional failures remain a major challenge and occur in up to 40% of patients in the first two years after the treatment [Chajon et al. 2013]. By focusing on metabolic and morphological tissue properties, respectively, PET and CT modalities include complementary and synergistic information for cancerous lesion segmentation as well as tumor characteristics potentially relevant for patient outcome prediction and HPV status diagnosis, in addition to usual clinical variables (e.g., age, gender, treatment modality, etc.). Modern image analysis (radiomics, machine, and deep learning) methods must be developed and, more importantly, rigorously evaluated, in order to extract and leverage this information. That is why, HEad and neCK TumOR (HECKTOR) Lesion Segmentation, Diagnosis and Prognosis challenge has been introduced in last few years. 

Following the success of the first editions of the HECKTOR challenge from 2020 through 2022, this challenge will be presented at the 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2025 in Daejeon, South Korea. This year, three tasks are proposed where the participants can choose to participate in any or all tasks. Participants will train models on our provided datasets, submit predictions, and compete on different metrics and robustness. Deadlines and leaderboard details are on the [challenge website](https://hecktor25.grand-challenge.org/timeline/). 

- **Tasks:**  
  - **Task 1:** The automatic detection and segmentation of Head and Neck (H&N) primary tumors and lymph nodes in FDG-PET/CT images.
  - **Task 2:** The prediction of Recurrence-Free Survival (RFS) from the FDG-PET/CT images, available clinical information, and radiotherapy planning dose maps.
  - **Task 3:** The diagnosis of HPV status from the FDG-PET/CT images and available clinical information.
- **Validation Submission Deadline:** 10th July to 14th August 2025
- **Testing Submission Deadline:** 15th August to 1st September 2025  
- **Website & Rules:** [Participation policies](https://hecktor25.grand-challenge.org/participation-policies/)


---

<!-- ## 🌲 Repository Branches

- **main**  
  - For **newcomers**: step-by-step guides, data loaders, training & inference scripts.  
  - **Goal:** get a working model up and running in minutes.  

- **docker-template**  
  - For **submission**: contains Docker setup, inference entrypoint, and all submission restrictions.  
  - **Goal:** containerize your final model for evaluation.  

--- -->

# 📑 Table of Contents

1. [Getting the Data](#-getting-the-data)
2. [Task Folders & Structure](#task-folders--structure)
3. [Environment Setup](#environment-setup)
4. [Training Your Model](#-training-your-model)  
5. [Inference & Evaluation](#-inference--evaluation)  
6. [Next Steps & Tips](#-next-steps--tips)  

---

# 📥 Getting the Data

1. **Download:** Go to the [Dataset Section](https://hecktor25.grand-challenge.org/dataset/) on challenge website and follow the instructions provided to download the dataset. 

2. **Dataset Structure:** Following is the structure of the dataset directory:

```text
hecktor2025_training/
  ├── Task 1
      ├── CHUM-001
        ├── CHUM-001__CT.nii.gz 
        ├── CHUM-001__PT.nii.gz
        └── CHUM-001.nii.gz # Label file (GTVp=1, GTVn=2)
      ├── CHUM-002
      ├── ...
      └── HECKTOR_2025_Training_Task_1.csv #Clinical data
  ├── Task 2
      ├── CHUM-001
        ├── CHUM-001__CT.nii.gz
        ├── CHUM-001__PT.nii.gz
        ├── CHUM-001__CTPlanning.nii.gz* # Subset only
        └── CHUM-001__RTDOSE.nii.gz* # Subset only
      ├── CHUM-002
      ├── ...
      └── HECKTOR_2025_Training_Task_2.csv # RFS endpoint data
  └── Task 3
      ├── CHUM-001
        ├── CHUM-001__CT.nii.gz
        └── CHUM-001__PT.nii.gz
      ├── CHUM-002
      ├── ...
      └── HECKTOR_2025_Training_Task_3.csv # HPV Status data
```
3. **Dataset Description**: The data originates from FDG-PET and low-dose non-contrast-enhanced CT images (acquired with combined PET/CT scanners) of the Head & Neck region. It was collected from 10 different [centers](https://hecktor25.grand-challenge.org/dataset/#dataset-structure). Following are the different formats of the dataset:

- **Image Data (PET/CT):**
  - All tasks include PET and CT scans for each patient, using the naming convention:
  - CenterName_PatientID__Modality.nii.gz
  - __CT.nii.gz — Computed tomography image
  - __PT.nii.gz — Positron emission tomography image
- **Segmentations (Task 1 only):**
  - Each patient has a single label file: PatientID.nii.gz
    - Label 1 = Primary tumor (GTVp)
    - Label 2 = Lymph nodes (GTVn)
- **Radiotherapy Dose Data (Task 2 only):**
  - For a subset of patients:
    - __CTPlanning.nii.gz — CT planning scan
    - __RTDOSE.nii.gz — RT dose map
- **Clinical Information:**
  Provided in HECKTOR_2025_Training_Task_#.csv, includes:
  - Center, gender, age, tobacco and alcohol use, performance status, treatment (radiotherapy only or chemoradiotherapy), M-stage (metastasis)
  - Relapse indicator and RFS value (used as the target for Task 2)
  - HPV status (used as the target for Task 3)

If you require any further details about the dataset, please visit the [Dataset](https://hecktor25.grand-challenge.org/dataset/#dataset-structure) section on the challenge website. 


---
# 🗂️ Task Folders & Structure

Each task folder is self-contained and contains only the scripts needed for that specific task. The finalized layout is:

```text
├── Task1/
│   ├── config/                   # Contains configuration files
│   ├── evaluation/               # inference_evaluator.py to compute metrics
│   ├── models/                   # supporting files for different models
│   ├── scripts/                  # training and inference scripts
│   ├── utils/                    # Shared helper functions (input/output, logging, visualization tools, etc.)
│   ├── README.md                 # Task1-specific README that explains how to build/run for Task 1
│   └── requirements.txt          # dependencies for Task 1
├── Task2/
│   ├── task2_prognosis.py        # Model training & evaluation for Task 2 (Prognosis)
│   └── task2_inference.py        # Inference entry-point for Task 2
└── Task3/
    ├── task3_classification.py   # Model training & evaluation for Task 3 (Classification)
    └── task3_inference.py        # Inference entry-point for Task 3
```
- **Task1/**
  - **scripts/train.py**: Train a segmentation model for Task 1.
  - **scripts/inference.py**: Inference script for Task 1 segmentation model.
  <!-- - **Usage**:
    ```bash
    # Perform Training
    python scripts/train.py --config unet3d
    ``` -->

- **Task2/**

  - **task2\_prognosis.py**: end-to-end training and validation script for Task 2’s prognosis model.
  - **task2\_inference.py**: Inference script for survival prediction using ensemble model.
  <!-- - **Usage**: 
    ```bash
    python inference_script.py \
    --csv test_data.csv --images_dir ./test_images \
    --ensemble ensemble_model.pt \
    --clinical_preprocessors  hecktor_cache_clinical_preprocessors.pkl
    ``` -->

- **Task3/**

  - **task3\_classification.py**: end-to-end training and validation script for Task 3’s classification model.
  - **task3\_inference.py**: Inference script for HPV status prediction used a trained model.
  <!-- - **Usage**: 
    ```bash
    python inference_classification.py \
    --input_path /input \
    --output_path /output \
    --scaler_file resources/scaler.joblib \
    --ohe_file resources/ohe.joblib \
    --checkpoint resources/best_model.pt \
    --ehr_file /input/ehr.json
  ``` -->

---
> **Baseline Notice:**
> This structure and the sample scripts are provided as a **baseline** to help you get started. You are **not required** to follow this exact layout or use the provided models. 

---
# ⚙️ Environment Setup

1. **Checkout main branch**

   ```bash
   git clone https://github.com/BioMedIA-MBZUAI/HECKTOR2025.git
   cd HECKTOR2025
   git checkout main
   ```
2. **Create virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install global requirements**

   ```bash
   pip install -r requirements.txt
   ```

---

# 🎯 Training Your Model

For each **TaskX/**, please run the training command accordingly as given below:
#### Task 1
  ```bash
  cd Task1/
  # Train for unet3d model
  python scripts/train.py --config unet3d 
  ```


#### Task 2
  ```bash
  cd Task2/
  # Train for 15 iterations
  python task2_prognosis.py 
  ```

#### Task 3
  ```bash
  cd Task3/
  # Train for 10 iterations
  python task3_classification.py 
  ```


<!-- * **Arguments** (`train.py`)

  * `--data-dir`: path to `data/`
  * `--save-dir`: where to write checkpoints
  * `--lr`, `--batch-size`, `--epochs`, etc. -->

<!-- * **Output:**

  * Trained weights in `checkpoint/`
  * Training logs printed to console -->

---


# 🔍 Inference & Evaluation

To run inference on validation data, use the below command accordingly for each task:

#### **Task 1**:
  ```bash
  cd Task1/
  python scripts/inference.py \
      --model_path best_model.pth \
      --ct_path /path/to/ct.nii.gz \
      --pet_path /path/to/pet.nii.gz \
      --output_path /path/to/output
  ```

#### **Task 2**:
  ```bash
  cd Task2/
  python python task2_inference.py \
    --csv test_data.csv \
    --input_path ./test_images \
    --ensemble ensemble_model.pt \
    --clinical_preprocessors  hecktor_cache_clinical_preprocessors.pkl \
  ```

#### **Task 3**:
  ```bash
  cd Task3/
  python task3_inference.py \
  --input_path /input \
  --output_path /output/sample_001_out.json \
  --scaler_file resources/scaler.joblib \
  --ohe_file resources/ohe.joblib \
  --checkpoint best_model.pt \
  --ehr_file /input/ehr.json
```

---

# 🌟 Next Steps & Tips

* **Data Augmentation:** Explore and try more aggressive transformations on the dataset.
* **Model Architecture:** Swap in a stronger backbone.
* **Hyperparameter Tuning:** Adjust learning rates, optimizers, schedulers.
* **Ensembling:** Combine outputs from multiple checkpoints.
* **Semi-Supervised Learning:** Leverage unlabelled test data for pseudo-labeling.

---



<!-- 


## Table of Contents
1. [Basic Instructions](#basic_instructions)
2. [Creating Docker Container](#creating_docker)
3. [Submitting Docker Container](#submitting_docker)
4. [Video Tutorial](#video_tutorial) -->


<!-- ## Creating Docker Container <a name="creating_docker"></a>

In this repository, you can find a template for creating valid a docker container for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). We also provide one example algorithm that has been prepared based on the aforementioned template: 

- `HECKTOR_template`: This directory contains a template with all the essential functions and modules needed to create an acceptable docker container for submitting to the HECKTOR challenge. Almost all of the functions and instructions in this template should remain the same and you just need to add/link your algorithm and weight files to them.
- `HECKTOR_baseline`: This directory contains a sample algorithm that has been prepared based on the aforementioned instructions within `HECKTOR_template`. For this example, the algorithm is the [To decide on whether to Direct To A folder Or Repo](https://github.com/vqdang/hover_net/tree/conic).

Each of these directories is accompanied by a `README.md` file in which we have thoroughly explained how you can dockerize your algorithms and submit them to the challenge. The code in the `HECKTOR_template` has been extensively commented and users should be able to embed their algorithms in the blank template, however, `HECKTOR_baseline` can be a good guide (example) to better understand the acceptable algorithm layout. 


 -->


<!-- ## Video Tutorial <a name="video_tutorial"></a>
For more information, please have a look at our [tutorial video (TBA)](https://conic-challenge.grand-challenge.org/Submission/). -->

---
# 📚 References

- [Bonner et al. 2010] Bonner, James A., Paul M. Harari, Jordi Giralt, Roger B. Cohen, Christopher U. Jones, Ranjan K. Sur, David Raben, et al. 2010. “Radiotherapy plus Cetuximab for Locoregionally Advanced Head and Neck Cancer: 5-Year Survival Data from a Phase 3 Randomised Trial, and Relation between Cetuximab-Induced Rash and Survival.” The Lancet Oncology 11 (1): 21–28.

- [Chajon et al. 2013] Chajon E, et al. "Salivary gland-sparing other than parotid-sparing in definitive head-and-neck intensity-modulated radiotherapy does not seem to jeopardize local control." Radiation Oncology 8.1 (2013): 1-9.

- [Parkin et al. 2005] Parkin DM, et al. "Global cancer statistics, 2002." CA: a cancer journal for clinicians 55.2 (2005): 74-108.


---
<div align="center">
  _You’re now ready to dive in and start building your own models.!_  
</div>
