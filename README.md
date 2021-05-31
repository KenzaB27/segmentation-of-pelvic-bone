# Segmentation and analysis of pelvic bone in CT images
In this repo, we address the problem of segmentation and analysis of pelvic bone in CT images in order to facilitate hip surgery planning. More specifically, we apply atlas based segmentation to localize the right femur and right hip bone in 3 common images. Evaluation is performed by the means of Dice Coefficient and Hausdorff distance. We also design a deep learning solution in order to detect the particular axial slice of a CT image which is most likely to contain the pubic symphysis. 

This is a project of the course HL2027 3D image reconstruction and analysis in medicine @ KTH (Royal Institute of Technology).

Authors: *Julia Onerud, Kenza Bouzid.*

## Environment

The project is implemented with SimpleITK for the atlas segmentation part and Tensorflow 2 for classification. 

Prepare an virtual environment with python>=3.6, and then use the following command line for the dependencies.

```bash
pip install -r requirements.txt
```

## Data 

The  experiments were conducted on CT scans provided by the course responsable.

Given three 3d CT scans manually segmented by an expert radiologist denoted as D_c (for common data) and three 3d CT scans specific to our group denoted as D_g, we aim to create atlases of D_g that can further on be used for multi-atlas based segmentation of D_c using registration techniques. 

D_g is used as training set for pubic symphysis detection. Testing is performed on D_c.

## Project structure

* The directory structure of the whole project is as follows:

```bash
│   .gitignore
│   README.md
│   requirements.txt
│
├───data
│
├───notebooks
│       atlas_seg_results.ipynb
│       atlas_segmentation.ipynb
│       data_preparation.ipynb
│       densenet.ipynb
│       inception.ipynb
│       ml_model.ipynb
│       resnet.ipynb
│       vgg.ipynb
│
└───src
    │   analysis.py
    │   registration.py
    │   segmentation.py
    └───utils.py
```

## References 

* Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, "SimpleITK Image-Analysis Notebooks: a Collaborative Environment for Education and Reproducible Research", J Digit Imaging., https://doi.org/10.1007/s10278-017-0037-8, 2017.

