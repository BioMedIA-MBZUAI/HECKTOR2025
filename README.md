# HECKTOR2025

<p align="center">
  <img src="/doc/images/HECKTOR-main.jpeg">
</p>

This repository contains instructions and examples for creating a valid docker for [HECKTOR 2025 Challenge](https://hecktor25.grand-challenge.org/hecktor25/) and how you can submit it to the [Grand Challenge](https://hecktor25.grand-challenge.org/hecktor25/) for evaluation.

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


Assuming you have a verified Grand-Challenge account and have already registered for the HECKTOR challenge, you need to do two main steps to submit your algorithm to the challenge. First, you need to [upload the algorithm](#uplaod-your-algorithm) docker container to the Grand-Challenge platform. Then, you can make a [submit that algorithm](#submit-your-algorithm) to compete in any leaderboard or phases of the challenge. But before you proceed, make sure that you have read and understood the [participation policies](https://hecktor25.grand-challenge.org/participation-policies/).

### 1- Upload your algorithm
> **IMPORTANT:** It is crucial to know that you have to submit different algorithms for different tasks of the challenge. Even if you are using the same method for all tasks, you have to upload your algorithm again because the Input and Output configurations for each tasks are different.

In order to submit your algorithm, you first have to add it to the Grand-Challenge platform. To do so, you have to follow the following steps: First, navigate to the [algorithm submission webpage](https://grand-challenge.org/algorithms/) and click on the "+ Add new algorithm" botttom:

<p align="center">
  <img src="/doc/images/Add-Algorithm.png">
</p> 

Then you will be directed to the "Create Algorithm" page where you have to fill some necessay fileds, as described below (please pay special attention to the files **Inputs** and **Outputs**):
## TODO: Change the images as currently the access was not provided.

<p align="center">
  <img src="/doc/algorithm fields.JPG">
</p>

- Title: The title of your algorithm to be shown on the leaderboard and your dashboard.
- Contact Email: The email of the person responsible for this algorithm (This email will be listed as the contact email for the algorithm and will be visible to all users of Grand Challenge.)
- Display Editors: Should the editors of this algorithm be listed on the information page. Preferably selected "Yes".
- Logo: Uploading a small logo image is mandatory by the grand-challenge platform. Try using an image that represents your team or algorithm.
- Viewer: This selects the viewer that might be used for showing algorithm results. Please select: "Viewer CIRRUS Core (Public)".


<p align="center">
<img src="/doc/task1_input_output.JPG">
</p>

<p align="center">
<img src="/doc/task2_input_output.JPG">
</p>

- **Inputs**: The input interface to the algorithm. This field determines what kind of input file the algorithm container expects to see in the input. (TODO: Verify the Inputs/Outputs from Shahd/Salma) For all the tasks of the challenge set this to be **Generic Medical Image (Image)**
- **Outputs**: The output interfaces for the algorithm. This field specifies the type of output(s) that algorithm generates. Please note that the output types are different for each task:
   1. **Task 1**: The "Outputs" field should be set to **Generic Medical Image (Image)**.
   2. **Task 2**: (TODO: Change the output of Task 2 as well accordingly.)
   3. **Task 3**: (TODO: Change the output of Task 3 as well accordingly.)



- Credits per job: keep this to 0.
- Image requires gpu: make sure to enable (check) the use of GPU if your algorithm needs one.
- Image requires memory gb: Specify how much RAM your algorithm requires to run. The maximum amount allowed is 32.

Once you have completed these required fields, press the "**Save**" button at the bottom of the page to create the algorithm and direct to the algorithm page where you can see the information regarding your algorithm and change them using "**Update Setting**" button if needed. Before you can use this algorithm for a challenge submission, you have to assign/upload your dockerized algorithm to it. To do so, click on the "**Containers**" tab from the left menu (TODO: To change this image.):

<p align="center">
<img src="/doc/algorithm_page.JPG">
</p>

Then, you have to click on the ![Upload Container](/doc/upload_container_botton.JPG) to navigate to the page where you can upload the packaged (compressed) docker container:

<p align="center">
<img src="/doc/container_upload.JPG">
</p>

Once you have uploaded your docker container and set the "GPU Supported" and "Requires memory gb" options (as explained before), click on the "Save" button and your algorithm will be completed and ready to be submitted to the challenge. Remember, the algorithm is only ready to submit when the status badge in front of the upload description changes to "Active".

### 2- Submit your algorithm
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
