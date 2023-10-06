# GUI to preprocess CTA images in 3D Slicer

This repository provides a GUI to automatically preprocess CTA images and evaluate their quality with 3D Slicer. 
This can be directly used as input for the ML pipeline to segment acute hypoperfused area indicative of ischemic stroke core available here: https://glabapps.uth.edu/cta/
and described here:
``
Giancardo L, Niktabe A, Ocasio L, Abdelkhaleq R, Salazar-Marioni S, Sheth SA. Segmentation of acute stroke infarct core using image-level labels on CT-angiography. NeuroImage: Clinical 2023;37:103362. https://doi.org/10.1016/j.nicl.2023.103362 . 
``




## Installation

1. Clone this repository
2. Open 3D Slicer (see https://www.slicer.org/ for installation instructions)
3. Open the Python Interactor
4. Run `exec(open('path/to/install_requirements.py').read())` in the Python interactor
5. 3D Slicer will reboot

## To Run

1. Open 3D Slicer
2. Open the Python Interactor
3. Run `import os; os.chdir('path/to/this/repository')`
4. Run `exec(open('gui_main.py').read())` in the Python interactor

# GPU/CPU configuration
The file conf.py contains the GPU/CPU configuration. This is needed for the rough brain mask detection, which is based on a 2D Attention U-Net. However, the whole process can be run pretty fast on CPU only. 