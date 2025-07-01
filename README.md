# HECKTOR2025 - Challenge

<p align="center">
  <img src="/doc/images/HECKTOR-main.jpeg">
</p>

Welcome to the **HECKTOR 2025 Challenge** repository! This repository contains instructions and examples for creating a baseline and a valid docker for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). It will also help you with how you can submit your designed model to the [Grand Challenge](https://hecktor25.grand-challenge.org/hecktor25/) for evaluation. Here you’ll find everything you need to get started quickly: from understanding the challenge, to setting up your environment, training your first model, and evaluating your results. So this reporsitory has **two primary branches** 🌲:

- [**main**](https://github.com/BioMedIA-MBZUAI/HECKTOR2025/tree/main): Here you’ll find step-by-step guides, data loaders, training scripts, and inference examples so you can get a working model up and running in minutes.

- [**docker-template**](https://github.com/BioMedIA-MBZUAI/HECKTOR2025/tree/docker-template): Designed for containerizing and submitting your final models to the Grand Challenge. This branch provides a Docker-based inference template, build/test/save scripts, and enforces all challenge restrictions (no network, CPU-only, `/tmp/` I/O, time limits, etc.).
---

# How can this Repo help?
This branch is the first thing you see when you click our challenge Github link. It's totally perfect if you’re new to the challenge or unfamiliar with our setup. You’ll find everything here to:

```text
1. Understand what the challenge is about  
2. Set up your development environment  
3. Train models on our provided data  
4. Test and evaluate your results  
5. Explore ideas for improving performance 
```

---
# 🚀 About the HECKTOR'25 Challenge
Head and Neck (H&N) cancers are among the most common cancers worldwide (5th leading cancer by incidence) [Parkin et al. 2005]. Radiotherapy combined with cetuximab has been established as a standard treatment [Bonner et al. 2010]. However, locoregional failures remain a major challenge and occur in up to 40% of patients in the first two years after the treatment [Chajon et al. 2013]. By focusing on metabolic and morphological tissue properties, respectively, PET and CT modalities include complementary and synergistic information for cancerous lesion segmentation as well as tumor characteristics potentially relevant for patient outcome prediction and HPV status diagnosis, in addition to usual clinical variables (e.g., age, gender, treatment modality, etc.). Modern image analysis (radiomics, machine, and deep learning) methods must be developed and, more importantly, rigorously evaluated, in order to extract and leverage this information. That is why, HEad and neCK TumOR (HECKTOR) Lesion Segmentation, Diagnosis and Prognosis challenge has been introduced in last few years. 

Following the success of the first editions of the HECKTOR challenge from 2020 through 2022, this challenge will be presented at the 28th International Conference on Medical Image Computing and Computer-Assisted Intervention (MICCAI) 2025 in Daejeon, South Korea. This year, three tasks are proposed where the participants can choose to participate in any or all tasks. Participants will train models on our provided datasets, submit predictions, and compete on different metrics and robustness. Deadlines and leaderboard details are on the [challenge website](https://hecktor25.grand-challenge.org/timeline/). 

- **Tasks:**  
  - **Task 1:** The automatic detection and segmentation of Head and Neck (H&N) primary tumors and lymph nodes in FDG-PET/CT images.
  - **Task 2:** The prediction of Recurrence-Free Survival (RFS) from the FDG-PET/CT images, available clinical information, and radiotherapy planning dose maps.
  - **Task 3:** The diagnosis of HPV status from the FDG-PET/CT images and available clinical information.
- **Submission Deadline:** 15th August to 1st September 2025  
- **Website & Rules:** [Participation Policies](https://hecktor25.grand-challenge.org/participation-policies/)


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
2. [Task Folders & Structure](#-task-folders--structure)  
3. [Environment Setup & Baseline](#-environment-setup--baseline)  
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
      └── HECKTOR_2025_Training_Task_1.csv # HPV Status data
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
  - Provided in HECKTOR_2025_Training_Task_#.csv, includes:
  - Center, gender, age, tobacco and alcohol use, performance status, treatment (radiotherapy only or chemoradiotherapy), M-stage (metastasis)
  - Relapse indicator and RFS value (used as the target for Task 2)
  - HPV status (used as the target for Task 3)
  - Some entries may contain missing data, but the 2025 edition includes significant updates.

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
  - **scripts/train.py**: Train a segmentation model for Task 1 (Available models: unet3d, segresnet, unetr, swinunetr)
  - **scripts/inference.py**: Evaluation script for Task 1 segmentation model.
  <!-- - **Usage**:
    ```bash
    # Perform Training
    python scripts/train.py --config unet3d
    ``` -->

- **Task2/**

  - **task2\_prognosis.py**: end-to-end training and validation script for Task 2’s prognosis model.
  - **task2\_inference.py**: Inference script for HECKTOR survival prediction using ensemble model.
  <!-- - **Usage**: 
    ```bash
    python inference_script.py \
    --csv test_data.csv --images_dir ./test_images \
    --ensemble ensemble_model.pt \
    --clinical_preprocessors  hecktor_cache_clinical_preprocessors.pkl
    ``` -->

- **Task3/**

  - **task3\_classification.py**: end-to-end training and validation script for Task 3’s classification model.
  - **task3\_inference.py**: loads saved weights and runs inference on a single sample or batch.
  <!-- - **Usage**: 
    ```Will need to add command for this Task as well.``` -->


Each script is ready to run out of the box. Just point it at your data directory and checkpoint folder to get started experimenting on that task.

---
> **Baseline Notice:**
> This structure and the sample scripts are provided as a **baseline** to help you get started. You are **not required** to follow this exact layout or use the provided models. Feel free to reorganize files, swap in your own approaches, or design your own workflow that best suits your development style.

---
# ⚙️ Environment Setup & Baseline

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
   pip install -r requirements-global.txt (If we have any, else can remove)
   ```
4. **Per-task dependencies (If opt. for separate Req.txt file)**

   ```bash
   cd Task1
   pip install -r requirements.txt
   cd ../Task2 && pip install -r requirements.txt
   cd ../Task3 && pip install -r requirements.txt
   ```
5. **Baseline Models**

    - Pretrained checkpoints live under each `/tmp/AddPath`.
    - They are already loaded in the baseline files for each task but you can also load by giving the path as an argument.

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
  # Inference on entire directory
  python scripts/inference.py \
      --model_path experiments/unet3d/checkpoints/best_model.pth \
      --input_dir /path/to/test/images \
      --output_dir /path/to/predictions

  # Inferenceo on single case
  python scripts/inference.py \
      --model_path experiments/unet3d/checkpoints/best_model.pth \
      --single_case \
      --ct_path /path/to/ct.nii.gz \
      --pet_path /path/to/pet.nii.gz \
      --output_dir /path/to/output
  ```

#### **Task 2**:
  ```bash
  cd Task2/
  python python inference_script.py \
    --csv test_data.csv \
    --images_dir ./test_images \
    --ensemble ensemble_model.pt \
    --clinical_preprocessors  hecktor_cache_clinical_preprocessors.pkl \
    --model-path checkpoint/best.pth \
    --input-path ../data/test/sample_001.png \
    --output-path results/sample_001_out.json
  ```

#### **Task 3**:
  ```bash
  cd Task3/
  Will need to add command for this Task as well.
  ```

- **Metrics:** We use **IoU**, **F1-score**, and **Accuracy**. (**Should we add the metrics**)

---

# 🌟 Next Steps & Tips

* **Data Augmentation:** Explore and try more aggressive transformations on the dataset.
* **Model Architecture:** Swap in a stronger backbone (ResNet → Swin Transformer).
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


# Basic Instructions <a name="basic_instructions"></a>




- This section is to guide the participants to the [submission tips](https://grand-challenge.org/documentation/making-a-challenge-submission/#submission-tips) documentation. This includes important information like:
  1. The duration of inference jobs will be limited by the challenge you created the container.
  2. The container you upload for your algorithm cannot exceed 10GB.
  3. You do not need to create a new algorithm for each submission.
  4. If you update your algorithm, don't forget to make a new submission to the challenge with it, this will not happen automatically. For more guidelines on how to creat a submission on Grand-Challenge and upload your algorithm, plese follow the intructions [here](/doc/submission-guidelines.md)
  5. If you test your own container you can see error messages the Results page of your algorithm. When running a sanity check we will provide the error messages for them.

- This is a [demo starter kit](https://github.com/DIAGNijmegen/demo-challenge-pack/tree/main?tab=readme-ov-file#now-what) that includes steps to be taken to get started with the challenge pack effectively.

<!-- ## Creating Docker Container <a name="creating_docker"></a>

In this repository, you can find a template for creating valid a docker container for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). We also provide one example algorithm that has been prepared based on the aforementioned template: 

- `HECKTOR_template`: This directory contains a template with all the essential functions and modules needed to create an acceptable docker container for submitting to the HECKTOR challenge. Almost all of the functions and instructions in this template should remain the same and you just need to add/link your algorithm and weight files to them.
- `HECKTOR_baseline`: This directory contains a sample algorithm that has been prepared based on the aforementioned instructions within `HECKTOR_template`. For this example, the algorithm is the [To decide on whether to Direct To A folder Or Repo](https://github.com/vqdang/hover_net/tree/conic).

Each of these directories is accompanied by a `README.md` file in which we have thoroughly explained how you can dockerize your algorithms and submit them to the challenge. The code in the `HECKTOR_template` has been extensively commented and users should be able to embed their algorithms in the blank template, however, `HECKTOR_baseline` can be a good guide (example) to better understand the acceptable algorithm layout. 


 -->


<!-- ## Video Tutorial <a name="video_tutorial"></a>
For more information, please have a look at our [tutorial video (TBA)](https://conic-challenge.grand-challenge.org/Submission/). -->

---
<div align="center">
  _You’re now ready to dive in and start building your own models.!_  
</div>