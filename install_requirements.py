'''
# install required packages using the Python interactor in 3D Slicer
# for example run as:
exec(open('THIS_FILE_NAME.py').read())

in my case:
exec(open('/data-full/giancardo-group/lgiancardo/slicer_scripts/preproc_cta_core/install_requirements.py').read())



'''
import os

import slicer
import slicer.util



#install
slicer.util.pip_install("torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
slicer.util.pip_install("pandas")
slicer.util.pip_install("monai")
slicer.util.pip_install("numpy")

# restart slicer
slicer.util.restart()