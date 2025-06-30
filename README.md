# HECKTOR2025 - Challenge

<p align="center">
  <img src="/doc/images/HECKTOR-main.jpeg">
</p>

Welcome to the **HECKTOR 2025 Challenge** repository! This repository contains instructions and examples for creating a baseline and a valid docker for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). It will also help you with how you can submit your designed model to the [Grand Challenge](https://hecktor25.grand-challenge.org/hecktor25/) for evaluation. Here you’ll find everything you need to get started quickly: from understanding the challenge, to setting up your environment, training your first model, and evaluating your results. So this reporsitory has **two primary branches** 🌲:

- **main**:  
  Here you’ll find step-by-step guides, data loaders, training scripts, and inference examples so you can get a working model up and running in minutes.

- **docker-template**:  
  Designed for containerizing and submitting your final models to the Grand Challenge. This branch provides a Docker-based inference template, build/test/save scripts, and enforces all challenge restrictions (no network, CPU-only, `/tmp/` I/O, time limits, etc.).
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
## 🚀 About the HECKTOR'25 Challenge
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

## 📑 Table of Contents

1. [Getting the Data](#-getting-the-data)  
2. [Task Folders & Structure](#-task-folders--structure)  
3. [Environment Setup & Baseline](#-environment-setup--baseline)  
4. [Training Your Model](#-training-your-model)  
5. [Inference & Evaluation](#-inference--evaluation)  
6. [Next Steps & Tips](#-next-steps--tips)  

---

## 📥 Getting the Data

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





























































































## Table of Contents
1. [Basic Instructions](#basic_instructions)
2. [Creating Docker Container](#creating_docker)
3. [Submitting Docker Container](#submitting_docker)
4. [Video Tutorial](#video_tutorial)


## Basic Instructions <a name="basic_instructions"></a>




- This section is to guide the participants to the [submission tips](https://grand-challenge.org/documentation/making-a-challenge-submission/#submission-tips) documentation. This includes important information like:
  1. The duration of inference jobs will be limited by the challenge you created the container.
  2. The container you upload for your algorithm cannot exceed 10GB.
  3. You do not need to create a new algorithm for each submission.
  4. If you update your algorithm, don't forget to make a new submission to the challenge with it, this will not happen automatically.
  5. If you test your own container you can see error messages the Results page of your algorithm. When running a sanity check we will provide the error messages for them.

- This is a [demo starter kit](https://github.com/DIAGNijmegen/demo-challenge-pack/tree/main?tab=readme-ov-file#now-what) that includes steps to be taken to get started with the challenge pack effectively.
- In case you need help in following a video, then [video1](https://www.youtube.com/watch?v=45BCMquFk70) and [video2](https://www.youtube.com/watch?v=Zkhrwark3bg) can be helpful which shows how to share your containers.


## Creating Docker Container <a name="creating_docker"></a>

In this repository, you can find a template for creating valid a docker container for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/). We also provide one example algorithm that has been prepared based on the aforementioned template: 

- `HECKTOR_template`: This directory contains a template with all the essential functions and modules needed to create an acceptable docker container for submitting to the HECKTOR challenge. Almost all of the functions and instructions in this template should remain the same and you just need to add/link your algorithm and weight files to them.
- `HECKTOR_baseline`: This directory contains a sample algorithm that has been prepared based on the aforementioned instructions within `HECKTOR_template`. For this example, the algorithm is the [To decide on whether to Direct To A folder Or Repo](https://github.com/vqdang/hover_net/tree/conic).

Each of these directories is accompanied by a `README.md` file in which we have thoroughly explained how you can dockerize your algorithms and submit them to the challenge. The code in the `HECKTOR_template` has been extensively commented and users should be able to embed their algorithms in the blank template, however, `HECKTOR_baseline` can be a good guide (example) to better understand the acceptable algorithm layout. 


## Submitting Docker Container <a name="submitting_docker"></a>


Assuming you have a verified Grand-Challenge account and have already registered for the HECKTOR challenge, you need to do two main steps to submit your algorithm to the challenge. First, you need to [upload the algorithm](#uplaod-your-algorithm) docker container to the Grand-Challenge platform. Then, you can follow the steps to [submit that algorithm](#submit-your-algorithm) to compete in any leaderboard or phases of the challenge. But before you proceed, make sure that you have read and understood the [participation policies](https://hecktor25.grand-challenge.org/participation-policies/).

### 1- Upload your algorithm <a name="uplaod-your-algorithm"></a>
> **IMPORTANT:** It is crucial to know that you have to submit different algorithms for different tasks of the challenge. Even if you are using the same method for all tasks, you have to upload your algorithm again because the Input and Output configurations for each tasks are different.

In order to submit your algorithm, you first have to add it to the Grand-Challenge platform. To do so, you have to follow the following steps: 

- First, navigate to the [algorithm submission webpage](https://grand-challenge.org/algorithms/) and click on the "+ Add new algorithm" botttom:

<p align="center">
  <img src="/doc/images/Add-Algorithm.png">
</p> 

- Then you will be directed to the "Create Algorithm" page where you have to choose the phase for which you are creating an algorithm using a drop-down list. Select the phase and click on "Create an Algorithm for this Phase". For more details on different phases, please click [here](#different_phases)

<p align="center">
  <img src="/doc/images/Phases.png">
</p> 

- The next step is to enter the the **Title** and **Job Description** for the algorithm. 

<p align="center">
  <img src="/doc/images/Algorithm-Details.png">
</p> 

> **NOTE:** Since you can only create a limited number of algorithms, please make the title of your algorithm meaningful and avoid titles that include the words "test", "debug" etc. In principle you will only need to create 1 algorithm for this phase. Once created, you can upload new container images for it as you improve your code and even switch back to older container images as you see fit.


- After successful creation of an algorithm, you have to connect the docker container. Now, before you can use this algorithm for a challenge submission, you have to assign/upload your dockerized algorithm to it. To do so, click on the "Containers" tab from the left menu.


<p align="center">
  <img src="/doc/images/Select-Containers.png">
</p> 


- There are two ways for uploading the containers. Either you need to link your algorithm to a **GitHub** repo and create a new tag, or upload a valid algorithm container image. 


<p align="center">
  <img src="/doc/images/Upload-Containers.png">
</p> 

- To upload a container, please make sure the file is in ```.tar.gz``` format produced from the command ```docker save IMAGE | gzip -c > IMAGE.tar.gz```. For more details, please see how to [save the container](https://docs.docker.com/engine/reference/commandline/save/). 

- The input/output for a container is also provided which determines what kind of input/outputs files are expected/generated by a container for each task. For more details on the specific format, you can click on the ℹ️ icon.

<p align="center">
  <img src="/doc/images/Input-Output-Container.png">
</p> 


- Once you have uploaded your docker container, click on the **Save** button given below and your algorithm will be completed and ready to be submitted to the challenge. Remember, the algorithm is only ready to submit when the status badge in front of the upload description changes to **Active**.

<p align="center">
  <img src="/doc/images/New-Container.png">
</p> 

## Submit Algorithm to be Added after Having some Discussion as it has some following issues which needs to be discussed.

<p align="center">
  <img src="/doc/images/Issue-Submission.png">
</p> 

### 2- Submit your algorithm <a name="different_phases"></a>
In the HECKTOR challenge, we have three tasks (`Task 1 - Detection and Segmentation`, `Task 2 - Prognosis`, and `Task 3 - Classification`) and for each task, participants compete in three phases. So here, the task submission is divided into 1+2 phases:

- **Sanity Check Phase:** Consists of 3 images to ensure participants are familiar with the Grand Challenge platform and that their dockers run without errors. All teams must make their submission to this phase and will receive feedback on any errors.
- **Validation Phase:** Consists of approximately 50 images. All teams will submit up to 2 working dockers from the sanity check to this phase. Only the top 15 teams, as ranked by the evaluation metrics displayed on the public validation leaderboard, with valid submissions will proceed to Phase 3.
- **Testing Phase:** Consists of approximately 400 images. The teams will choose 1 of their 2 dockers from the validation phase to be submitted to the testing phase. The official ranking of the teams will be based solely on the testing phase results.


> **NOTE:** The participants will not receive detailed feedback during the testing phase except for error notifications.

Below here are some of the requirements for the submission to be valid. 

- **Task 1:** Docker containers should produce segmentation outputs as a single label mask per patient (1 for the predicted GTVp, 2 for GTVn, and 0 for the background) in `.mha` format. The resolution of this mask should be the same as the original CT resolution. Participants should ensure correct pixel spacing and origin with respect to the original reference frame. The mha files should be named `[PatientID].mha`, matching the patient names, e.g., `CHUB-001.mha`.
- **Task 2:** Docker containers should produce results as a JSON file with the prediction output of the model as a float. The output should be anti-concordant with the RFS in days (i.e., the model should output a predicted risk score).
- **Task 3:** Docker containers should produce results as a JSON file containing the output of the model (boolean). The output should be True for HPV positive and False for HPV negative.

To start with your submission, for each task on either phases, you have to navigate to the challenge ["Submission" page (TBA)](https://hecktor25.grand-challenge.org/evaluation/challenge/submissions/create/):

<p align="center">
<img src="/doc/submissions.JPG">
</p>

On the top region, you can select for which phase and task you are submitting your method. Assuming that we want to test it on the **Validation phase**, we select the "**Detection and Segmentation - Validation Test**" tab.

<p align="center">
<img src="/doc/submit_algorithm.jpg">
</p>

The most important thing here is to select the algorithm you created for this task from the "**Algorithms**" list. You can also write comments about the submitted algorithm. Also, if you are submitting an algorithm for one of the tasks in the **Testing Phase**, it is mandatory to past a link to the `ArXiv` manuscript in which you have explained the technical details of your algorithm in the **Preprint (Algorithm Description)** field. Finally, by clicking on the "**Save**" button you will submit your algorithm for evaluation on the challenges task. The process is the same for all the tasks and phases.



## Video Tutorial <a name="video_tutorial"></a>
For more information, please have a look at our [tutorial video (TBA)](https://conic-challenge.grand-challenge.org/Submission/).
