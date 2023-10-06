'''
Run as

import os
os.chdir('/data-full/giancardo-group/lgiancardo/slicer_scripts/preproc_cta_core/')
exec(open('gui_main.py').read())
'''

import qt
import slicer
import os
import SimpleITK as sitk
import sitkUtils



# add cwd to path
import sys
sys.path.append(os.getcwd())

import inference

#=== Constants
IMG_TEMPLATE = 'res/coreTemplate-MNI152lin_T1_1mm.nii.gz'
IMG_TEMPLATE_MASK = 'res/coreTemplate-MNI152lin_T1_1mm-mask.nii.gz'

#===

def displayMessage(message):
    """Show a pop up message immediately."""
    qt.QMessageBox.information(slicer.util.mainWindow(), 'CTA Core', message)
    print(message)
    



def delayDisplay(message, msec=1000):
    """Show a pop up message after a delay.
    This may be useful to prevent the pop up from appearing
    while processing is still happening.
    """
    qt.QTimer.singleShot(msec, lambda: qt.QMessageBox.information(slicer.util.mainWindow(), 'CTA Core', message))


def setStatus(message):
    global labelStatusCurr
    labelStatusCurr.setText(message)
    slicer.app.processEvents()

def onModuleSelected(modulename):
    global tabWidget
    scrollArea = qt.QScrollArea(tabWidget)
    scrollArea.setWidget(getattr(slicer.modules, modulename.lower()).widgetRepresentation())
    scrollArea.setHorizontalScrollBarPolicy(qt.Qt.ScrollBarAlwaysOff)
    scrollArea.setWidgetResizable(True)
    tabWidget.addTab(scrollArea, modulename)


def genMask():
    global volumeSelector, volSelectorMask
    # get selected volume 
    volumeNode = volumeSelector.currentNode()
    if volumeNode is None:
        displayMessage("Please select a volume")
        return

    # get volume ID
    # volumeID = volumeNode.GetID()
    imgSi = sitkUtils.PullVolumeFromSlicer(volumeNode)
    
    # generate mask
    setStatus('Generating rough brain mask')
    resVolSi = inference.segmentBrainMask(imgSi)

    # print("Brain mask generated = ", resVolSi.GetSize())
    # add to scene
    volMask = sitkUtils.PushVolumeToSlicer( resVolSi, name='ctaMask' )

    # display message
    setStatus('Mask generated, please register to template')

    # set mask volume
    volSelectorMask.setCurrentNode(volMask)




def getMaskVolume():
    global volSelectorMask

    # get selected volume 
    volumeNode = volSelectorMask.currentNode()
    if volumeNode is None:
        displayMessage("Please generate a mask or select an existing mask volume")
        return None

    return volumeNode


def registerToTemplateWithTransform():
    global volumeSelector, templateVol, templateMaskLVol
    # get selected volume
    volumeNode = volumeSelector.currentNode()

    # get ctaMask by name
    ctaMaskNode = getMaskVolume()

    # slicer.util.selectModule('BRAINSFit')
    brainsFit = slicer.modules.brainsfit   

    linearTransform = slicer.vtkMRMLLinearTransformNode()
    linearTransform.SetName('regCtaTransform')
    slicer.mrmlScene.AddNode( linearTransform )


    #==== register preserving the orginal volume
    parametersRigid = {}
    parametersRigid["fixedVolume"] = templateVol.GetID()
    parametersRigid["movingVolume"] = volumeNode.GetID()
    parametersRigid["linearTransform"] = linearTransform.GetID()
    # parametersRigid["outputVolume"] = 'registered.nii.gz'
    parametersRigid["useRigid"] = True
    parametersRigid["initializeTransformMode"] = "useCenterOfROIAlign"
    parametersRigid["maskProcessingMode"] = "ROI"
    parametersRigid["fixedBinaryVolume"] = templateMaskLVol.GetID()
    parametersRigid["movingBinaryVolume"] = ctaMaskNode.GetID()
    
    parametersRigid["samplingPercentage"] = 0.02
    cliBrainsFitRigidNode = slicer.cli.run(brainsFit, None, parametersRigid)
    setStatus('Start Registering...')
    #====
    # detect if registration is done
    cliBrainsFitRigidNode.AddObserver('ModifiedEvent', 
                                      lambda caller, event: setStatus('Registration done') \
                                      if cliBrainsFitRigidNode.GetStatusString() == 'Completed' else None)



def registerToTemplateAndOut():
    global volumeSelector, templateVol, templateMaskLVol
    # get selected volume
    volumeNode = volumeSelector.currentNode()

    # get ctaMask by name
    ctaMaskNode = getMaskVolume()

    # slicer.util.selectModule('BRAINSFit')
    brainsFit = slicer.modules.brainsfit   

    # linearTransform = slicer.vtkMRMLLinearTransformNode()
    # linearTransform.SetName('regCtaTransform')
    # slicer.mrmlScene.AddNode( linearTransform )
    # create image node
    registeredImageNode = slicer.vtkMRMLScalarVolumeNode()
    registeredImageNode.SetName('registered')
    slicer.mrmlScene.AddNode( registeredImageNode )

    #==== register preserving the orginal volume
    parametersRigid = {}
    parametersRigid["fixedVolume"] = templateVol.GetID()
    parametersRigid["movingVolume"] = volumeNode.GetID()
    # parametersRigid["linearTransform"] = linearTransform.GetID()
    parametersRigid["outputVolume"] = registeredImageNode.GetID()
    parametersRigid["useRigid"] = True
    parametersRigid["initializeTransformMode"] = "useCenterOfROIAlign"
    parametersRigid["maskProcessingMode"] = "ROI"
    parametersRigid["fixedBinaryVolume"] = templateMaskLVol.GetID()
    parametersRigid["movingBinaryVolume"] = ctaMaskNode.GetID()
    
    parametersRigid["samplingPercentage"] = 0.02
    cliBrainsFitRigidNode = slicer.cli.run(brainsFit, None, parametersRigid)
    setStatus('Registering')
    #====

    def registrationDone(caller, event):
        if cliBrainsFitRigidNode.GetStatusString() == 'Completed':
            # apply mask
            setStatus('Registration done - Applying mask')
            registeredImageSi = sitkUtils.PullVolumeFromSlicer( registeredImageNode )
            templateMaskImageSi = sitkUtils.PullVolumeFromSlicer( templateMaskLVol )
            # apply mask
            maskFilter = sitk.MaskImageFilter()
            maskedSi = maskFilter.Execute(registeredImageSi, templateMaskImageSi )

            maskedNode= sitkUtils.PushVolumeToSlicer( maskedSi, name='registered_masked' )

            # set active volume and status
            slicer.util.setSliceViewerLayers(background=maskedNode)
            setStatus('Registered and template mask applied')


        

    # detect if registration is done
    # cliBrainsFitRigidNode.AddObserver('ModifiedEvent', 
    #                                   lambda caller, event: setStatus('Registration done') \
    #                                   if cliBrainsFitRigidNode.GetStatusString() == 'Completed' else None)
    cliBrainsFitRigidNode.AddObserver( 'ModifiedEvent', registrationDone )




#==== GUI START
mainWidget = qt.QWidget()
vlayout = qt.QVBoxLayout()
mainWidget.setLayout(vlayout)

# splitter between "layout" and "bottom frame" (not used)
splitter = qt.QSplitter()
splitter.orientation = qt.Qt.Vertical
vlayout.addWidget(splitter)


# Bottom frame: including control buttons and tab widget
bottomFrame = qt.QFrame()
bottomVlayout = qt.QVBoxLayout()
bottomFrame.setLayout(bottomVlayout)
splitter.addWidget(bottomFrame)

#==== Row 0
hlayout0 = qt.QHBoxLayout()
bottomVlayout.addLayout(hlayout0)

labelStatus0 = qt.QLabel("Current Status: ")
labelStatusCurr = qt.QLabel("Idle")
hlayout0.addWidget(labelStatus0)
hlayout0.addWidget(labelStatusCurr)
#====
#==== Row 1
# Horizontal layout
hlayout = qt.QHBoxLayout()
# add label
label1 = qt.QLabel("Input Volume: ")
hlayout.addWidget(label1)

# add volume selector
volumeSelector = slicer.qMRMLNodeComboBox()
volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
volumeSelector.selectNodeUponCreation = True
volumeSelector.addEnabled = False
volumeSelector.removeEnabled = False
volumeSelector.noneEnabled = False
volumeSelector.showHidden = False
volumeSelector.showChildNodeTypes = False
volumeSelector.setMRMLScene(slicer.mrmlScene)
volumeSelector.setToolTip("Input Volume")
hlayout.addWidget(volumeSelector)



bottomVlayout.addLayout(hlayout)

#====
#==== Row 2
# Horizontal layout
hlayout2 = qt.QHBoxLayout()

# add mask button
maskButton = qt.QPushButton("Generate Rough Brain Mask")
hlayout2.addWidget(maskButton)
maskButton.connect('clicked()', genMask)


# add label

hlayout2.addWidget(qt.QLabel("Current Mask: "))

# add volume selector
volSelectorMask = slicer.qMRMLNodeComboBox()
volSelectorMask.nodeTypes = ["vtkMRMLScalarVolumeNode"]
volSelectorMask.selectNodeUponCreation = True
volSelectorMask.addEnabled = False
volSelectorMask.removeEnabled = False
volSelectorMask.noneEnabled = False
volSelectorMask.showHidden = False
volSelectorMask.showChildNodeTypes = False
volSelectorMask.setMRMLScene(slicer.mrmlScene)
volSelectorMask.setToolTip("Mask Volume")
hlayout2.addWidget(volSelectorMask)






bottomVlayout.addLayout(hlayout2)

#====
#==== Row 3
#==== Show registration button
hlayout3 = qt.QHBoxLayout()
bottomVlayout.addLayout(hlayout3)


regTrButton = qt.QPushButton("Register With Transform")
regTrButton.connect('clicked()', registerToTemplateWithTransform)

reg2Button = qt.QPushButton("Register and Output Volume")
reg2Button.connect('clicked()', registerToTemplateAndOut)


hlayout3.addWidget(regTrButton)
hlayout3.addWidget(reg2Button)

#====

# moduleSelector = slicer.qSlicerModuleSelectorToolBar()
# moduleSelector.setModuleManager(slicer.app.moduleManager())
# hlayout.addWidget(moduleSelector)

# # Tab widget
# tabWidget = qt.QTabWidget()
# bottomVlayout.addWidget(tabWidget)


# # modules = ["volumes", "models", "labelstatistics"]
# modules = ["volumes", "models"]
# for module in modules:
#   onModuleSelected(module)

#==== GUI END

#== init volumes
try:
    # try load existing template image
    templateVol = slicer.util.getNode('template')
    displayMessage('Template image already loaded')
except slicer.util.MRMLNodeNotFoundException as e:
    # load template image
    templateVol = slicer.util.loadVolume(IMG_TEMPLATE, {'name':'template'})

try:
    # try load existing template image
    templateMaskLVol = slicer.util.getNode('templateMask')
    displayMessage('Template Mask image already loaded')
except slicer.util.MRMLNodeNotFoundException as e:
    # load template image
    templateMaskLVol = slicer.util.loadVolume(IMG_TEMPLATE_MASK, {'name':'templateMask'})


#==

# move window to top left
mainWidget.window().move(100,100)
# show window
mainWidget.show()
