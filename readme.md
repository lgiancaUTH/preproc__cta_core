# GUI to preprocess CTA images in 3D Slicer

This repository provides a GUI to automatically preprocess CTA images and evaluate their quality with 3D Slicer. 

This GUI allows to estimate a rough brain mask and register a CTA image to a template. It allows both to move the image to a new space (and anonimize it by applying a conservative brain mask) or to compute the affine matrix without chaging the original CTA. Note that the brain mask is not intended to include only the brain, but also the the internal edges of the skull. This is by design, as it allows fully capturing the circulation close to the skull.


The ouput can be directly used as input for the ML pipeline to segment acute hypoperfused area indicative of ischemic stroke core available here: https://glabapps.uth.edu/cta/
and described here:
``
Giancardo L, Niktabe A, Ocasio L, Abdelkhaleq R, Salazar-Marioni S, Sheth SA. Segmentation of acute stroke infarct core using image-level labels on CT-angiography. NeuroImage: Clinical 2023;37:103362. https://doi.org/10.1016/j.nicl.2023.103362 . 
``




## Installation

1. Clone this repository `git clone https://github.com/lgiancaUTH/preproc__cta_core.git` (or download and unzip the repository by clicking on the green button on the top right of this main GitHub repository page)
2. Open 3D Slicer (see https://www.slicer.org/ for installation instructions)
3. Open the Python Interactor
4. Run `exec(open('path/to/install_requirements.py').read())` in the Python interactor
5. 3D Slicer will reboot

## To Run

1. Open 3D Slicer
2. Open the Python Interactor
3. Run `import os; os.chdir('path/to/this/repository')`
4. Run `exec(open('gui_main.py').read())` in the Python interactor


## Usage
A video describing its use is available here: https://youtu.be/IQiWHLVHE4o

# GPU/CPU configuration
The file conf.py contains the GPU/CPU configuration. This is needed for the rough brain mask detection, which is based on a 2D Attention U-Net. However, the whole process can be run pretty fast on CPU only. 

# Contact
For more information, please contact us: https://sbmi.uth.edu/giancalab/ 

# Disclaimer
This tool is experimental and for research purposes only and is not intended for diagnostic use or medical decision-making. The tool  are prototypes and have not been approved by the FDA. 

# Acknowledgements
This work is supported by the NIH NINDS R01NS121154.
