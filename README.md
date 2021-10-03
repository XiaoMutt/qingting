# Introduction
This project experimented with reinforcement learning in object detection. The goal is to use reinforcement learning in 
cancer cell identification and segmentation of digital pathology images. Digital pathology images are extremely large 
(in hundreds of GB scale) and each image contains multiple zooming levels. To process such large images, they are often 
tessellated into small images which are sent to the ML/AI pipelines.

Instead of tessellating the image, this project's ultimate goal is to train the reinforcement learning agent to 
efficiently traverse different zooming levels and find the correct ROI. For one thing, this process mimics how 
pathologists process these images. For another thing, this is similar to playing an ROI-finding game in a "world" 
with levels.

As a first step attempt, and due to the computational power limitation, this project is simplified to find hand-written 
digits on grayscale daily-life photos (flickr30k). In brief, hand-written digits are synthesized into the photos, 
a reinforcement learning agent needs to move a square cursor to collect all the digits. Then the digits are sent to 
a downstream network for identification and segmentation.

For details see the report.pdf.

# TL;DR
## Model Used
- Reinforcement Learning Model: double DQN
- Identification and Segmentation Model: ResNet

## Folder Structure
- core: basic infra code for data ETL and ML
    - basis: basic components
    - etl: for data ETL (image conversion and synthesis)
    - ml: machine learning
        - dqn: the reinforcement learning model
        - ois: the identification and segmentation model
- scripts: script for model training and evaluation
- tests: test code

## Result
### Object Region Detection by the DQN Agent
After several rounds of training and testing, the best the DQN can do is moving the cursor close to the digit and
wandering around there. The DQN agent fails to collect the digit in most cases. I am not able to get further training 
and testing due to lack of AWS credits. Currently, all video cards are back-ordered, and training on my iMac is 
painfully slow. Therefore, this project is postponed.

### Object Identification and Segmentation
This part is a success. The ResNet-based model can reach high accuracy for both digit identification and segmentation.

# Code Referred:
- DQN: Stanford CS234 Reinforcement Learning homework2 with massive refactoring
- Rainbow: https://github.com/Curt-Park/rainbow-is-all-you-need