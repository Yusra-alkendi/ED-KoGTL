# Neuromorphic Camera Denoising using Graph Neural Network-driven Transformers

In this paper, we propose a novel offline methodology, referred to as Known-object Ground-Truth Labeling (KoGTL) which classifies DVS events stream into two main classes: real or noise event. 
The proposed KoGTL labeling algorithm is divided into three main stages including Event-Image Synchronization, Event-Edge Fitting and Event-Labeling as depicted in the following.




![MAINCOMPONENTSOFVISUAL-LOCALIZATION](https://github.com/Yusra-alkendi/EventDenoising_GNNTransformer/blob/2255aa7e3d25f7a0d91183c069412aa3ea8aafcf/KOGTL3.jpg)

## Known-object Ground-Truth Labeling (KoGTL) Framework


This is the code for the 2021 IEEE-access paper:

The main idea behind the KoGTL is to use a multi-trial experimental approach to record event streams and then perform labeling. More specifically, a dynamic active pixel vision sensor (DAVIS346C) is mounted on a Universal Robot UR10 6-DOF arm, in a front forward position and repeatedly moved along a certain (identical) trajectory under various illumination conditions.

The events are recorded along with two other measurements: (1) the camera pose at which the data was recorded, which we obtain through kinematics of the robot arm and (2) the intensity measurements from the scene obtained using the augmented active pixel sensor (APS images).
Several experimental scenarios are adopted where data is acquired at repeated transnational motion of the robot along square trajectory under different lighting conditions such as ∼750lux (Good light) and ∼5lux (low light). Streams of events with corresponding APS images and robot poses were acquired for about 5s per experimental scenario. 
## Event-Denoising-Labeled-Dataset- Files

Row experimental data:

**(1)** **"RawDVS_ExperimentalData":** raw sensor data is in ".mat" format. 

The labelled dataset for each experimental scenarios:

**(2)** **"Dataset_Goodlight_750lux":** contains labeled event dataset of ∼750lux (Good light).

**(3)** **"Dataset_Lowlight_5lux":** contains labeled event dataset of ∼5lux (Low light). 

For additional information please see the paper.


# Additional Notes
Feel free to contact the repository owner if you need any help with using the Labelled dataset <yusra.alkendi@ku.ac.ae>. 
