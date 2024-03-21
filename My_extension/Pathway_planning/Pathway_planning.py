import logging
import os
from typing import Annotated, Optional
import vtk
import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLMarkupsFiducialNode, vtkMRMLLabelMapVolumeNode
import warnings

# Pathway_planning
class Pathway_planning(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("Path planning")  # TODO: make this more human readable by adding spaces    
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "My_modules")] # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Dang The Hung (King's College London)"]  # TODO: replace with "Firstname Lastname (Organization)"     
        self.parent.helpText = _("""This is an extension to find a straight line that get to the target, at good angle, avoid and maximzing distance to obstacle, 
""") # _() function marks text as translatable to other languages
        self.parent.acknowledgementText = _("""Thank you!!!""")
#########################################################################################
        # # Additional initialization step after application startup is complete
        # slicer.app.connect("startupCompleted()", registerSampleData)

# Register sample data sets in Sample Data module. Local storage -> remove
def registerSampleData():
    """Add data sets to Sample Data module."""
    import SampleData
    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # Pathway_planning1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Pathway_planning",
        sampleName="Pathway_planning1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "Pathway_planning1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="Pathway_planning1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="Pathway_planning1",
    )

    # Pathway_planning2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="Pathway_planning",
        sampleName="Pathway_planning2",
        thumbnailFileName=os.path.join(iconsPath, "Pathway_planning2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="Pathway_planning2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="Pathway_planning2",
    )

# Pathway_planningParameterNode
@parameterNodeWrapper
class Pathway_planningParameterNode:
    entryPoints: vtkMRMLMarkupsFiducialNode
    targetPoints: vtkMRMLMarkupsFiducialNode
    targetLabelMap: vtkMRMLLabelMapVolumeNode
    obstacleLabelMap: vtkMRMLLabelMapVolumeNode
    softconstraintLabelMap: vtkMRMLLabelMapVolumeNode
    outputVolume: vtkMRMLLabelMapVolumeNode

# Pathway_planningWidget
class Pathway_planningWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        # Additional widgets can be instantiated manually and added to self.layout.
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/Pathway_planning.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = Pathway_planningLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
        self.ui.dataCleaning.connect("clicked(bool)", self.dataCleaning)
        self.ui.applyButton.connect("clicked(bool)", self.onApplyButton)

        # Make sure parameter node is initialized (needed for module reload)
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        # Make sure parameter node exists and observed
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        # Do not react to parameter node changes (GUI will be updated when the user enters into the module)
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        # Parameter node will be reset, do not use it anymore
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        # If this module is shown while the scene is closed then recreate a new parameter node immediately
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed."""
        # Parameter node stores all user choices in parameter values, node selections, etc.
        # so that when the scene is saved and reloaded, these settings are restored.

        self.setParameterNode(self.logic.getParameterNode())

        # Select default input nodes if nothing is selected yet to save a few clicks for the user
        if not self._parameterNode.entryPoints:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.entryPoints = firstVolumeNode

        if not self._parameterNode.targetPoints:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLMarkupsFiducialNode")
            if firstVolumeNode:
                self._parameterNode.targetPoints = firstVolumeNode

        if not self._parameterNode.targetLabelMap:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.targetLabelMap = firstVolumeNode

        if not self._parameterNode.obstacleLabelMap:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.obstacleLabelMap = firstVolumeNode

        if not self._parameterNode.softconstraintLabelMap:
            firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLLabelMapVolumeNode")
            if firstVolumeNode:
                self._parameterNode.softconstraintLabelMap = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[Pathway_planningParameterNode]) -> None:
        """
        Set and observe parameter node.
        Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
        """

        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
        self._parameterNode = inputParameterNode
        if self._parameterNode:
            # Note: in the .ui file, a Qt dynamic property called "SlicerParameterName" is set on each
            # ui element that needs connection.
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanApply)
            self._checkCanApply()

    def _checkCanApply(self, caller=None, event=None) -> None:
        if self._parameterNode and self._parameterNode.entryPoints:
            self.ui.applyButton.toolTip = _("Compute output volume")
            self.ui.applyButton.enabled = True
        else:
            self.ui.applyButton.toolTip = _("Select input and output volume nodes")
            self.ui.applyButton.enabled = False

        if self._parameterNode and self._parameterNode.entryPoints:
            self.ui.dataCleaning.enabled = True
        else:
            self.ui.dataCleaning.enabled = False

    def dataCleaning(self) -> None:
        """Run processing when user clicks "Summarize the data and create models from label maps" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.summarize_Data_and_Create_Model(self._parameterNode.entryPoints, self._parameterNode.targetPoints)

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process(self.ui.inputSelector.currentNode(), self.ui.outputSelector.currentNode(),
                               self.ui.imageThresholdSliderWidget.value, self.ui.invertOutputCheckBox.checked)

            # Compute inverted output (if needed)
            if self.ui.invertedOutputSelector.currentNode():
                # If additional output volume is selected then result with inverted threshold is written there
                self.logic.process(self.ui.inputSelector.currentNode(), self.ui.invertedOutputSelector.currentNode(),
                                   self.ui.imageThresholdSliderWidget.value, not self.ui.invertOutputCheckBox.checked, showResult=False)

# Pathway_planningLogic
class Pathway_planningLogic(ScriptedLoadableModuleLogic):
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()

    def getParameterNode(self):
        return Pathway_planningParameterNode(super().getParameterNode())

    def summarize_Data_and_Create_Model(self, entryNode, targetNode):
        # Get number of entry and target point
        n_Entry = entryNode.GetNumberOfControlPoints()
        n_Target = targetNode.GetNumberOfControlPoints()
        print(f"Number of entries is: {n_Entry}")
        print(f"Number of targets is: {n_Target}")
        # Create segmentation and model from label map
        list_of_Label_nodes = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        exportModelFolderItemId = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), "Models")
        exportSegmentFolderItemId = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), "Segments")
        for i, node in enumerate(list_of_Label_nodes):
            seg = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
            seg.SetName("Segmentation"+node.GetName())
            self.shNode.SetItemParent(self.shNode.GetItemByDataNode(seg), exportSegmentFolderItemId)
            slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(node, seg)
            slicer.modules.segmentations.logic().ExportAllSegmentsToModels(seg, exportModelFolderItemId)
            self.shNode.SetItemName(self.shNode.GetItemChildWithName(exportModelFolderItemId, "1"), "Model_"+ node.GetName())
        self.shNode.SetItemParent(exportModelFolderItemId, self.shNode.GetItemParent(exportSegmentFolderItemId))
        self.shNode.SetItemExpanded(exportSegmentFolderItemId, 0)

        return n_Entry, n_Target

    def process(self,
                inputVolume: vtkMRMLScalarVolumeNode,
                outputVolume: vtkMRMLScalarVolumeNode,
                imageThreshold: float,
                invert: bool = False,
                showResult: bool = True) -> None:
        """
        Run the processing algorithm.
        Can be used without GUI widget.
        :param inputVolume: volume to be thresholded
        :param outputVolume: thresholding result
        :param imageThreshold: values above/below this threshold will be set to 0
        :param invert: if True then values above the threshold will be set to 0, otherwise values below are set to 0
        :param showResult: show output volume in slice viewers
        """

        if not inputVolume or not outputVolume:
            raise ValueError("Input or output volume is invalid")

        import time

        startTime = time.time()
        logging.info("Processing started")

        # Compute the thresholded output volume using the "Threshold Scalar Volume" CLI module
        cliParams = {
            "InputVolume": inputVolume.GetID(),
            "OutputVolume": outputVolume.GetID(),
            "ThresholdValue": imageThreshold,
            "ThresholdType": "Above" if invert else "Below",
        }
        cliNode = slicer.cli.run(slicer.modules.thresholdscalarvolume, None, cliParams, wait_for_completion=True, update_display=showResult)
        # We don't need the CLI module node anymore, remove it to not clutter the scene with it
        slicer.mrmlScene.RemoveNode(cliNode)

        stopTime = time.time()
        logging.info(f"Processing completed in {stopTime-startTime:.2f} seconds")

# Pathway_planningTest
class Pathway_planningTest(ScriptedLoadableModuleTest):
    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.shNode.RemoveAllItems()
        self.logic = Pathway_planningLogic()
        self.test_data_folder_path = "D:\\Code\\3D_slicer\\Navigation_robotics\\Data\\Week_2_4_TestSet_Path_planning\\"

        # Declare folder for saving test model
        self.volumefortestfolderID = self.simple_Task_Create_Folder("VolumeForTest")
        self.labelvolumefortestfolderID = self.simple_Task_Create_Folder("LabelVolumeForTest")
        self.segmentationfortestfolderID = self.simple_Task_Create_Folder("SegmentationForTest")
        self.modelfortestfolderID = self.simple_Task_Create_Folder("ModelForTest")

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.load_Data()
        self.test_summarize_Data_and_Create_Model()
        self.object_Creation_For_Testing_Creating_Box_All()
        self.object_Creation_For_Testing_Creating_Box_Volume("Volume")
        self.object_Creation_For_Testing_Creating_Box_Label_Volume("Label")
        self.object_Creation_For_Testing_Creating_Box_Segmentation("Segmentation")
        self.object_Creation_For_Testing_Creating_Box_Model("Model")
        self.test_Check_A_Point_Is_In_A_Label_Volume_Node()
    def load_Data(self):
        # Load test data
        self.start_points = slicer.util.loadMarkups(self.test_data_folder_path + "entriesSubsample.fcsv")
        self.end_points = slicer.util.loadMarkups(self.test_data_folder_path+"targetsSubsample.fcsv")
        self.cortex_label_map = slicer.util.loadLabelVolume(self.test_data_folder_path+"r_cortexTest.nii.gz")
        self.hypothalamus_label_map = slicer.util.loadLabelVolume(self.test_data_folder_path+"r_hippoTest.nii.gz")
        self.ventricle_label_map = slicer.util.loadLabelVolume(self.test_data_folder_path+"ventriclesTest.nii.gz")
        self.vessel_label_map = slicer.util.loadLabelVolume(self.test_data_folder_path+"vesselsTestDilate1.nii.gz")
        self.general_volume = slicer.util.loadVolume(self.test_data_folder_path+"fakeBrainTest.nii.gz")
        # Hide the nodes
        pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
        volItem = self.shNode.GetItemByDataNode(self.general_volume)
        volPlugin = pluginHandler.getOwnerPluginForSubjectHierarchyItem(volItem)
        volPlugin.setDisplayVisibility(volItem, 0)
        volItem = self.shNode.GetItemByDataNode(self.vessel_label_map)
        volPlugin = pluginHandler.getOwnerPluginForSubjectHierarchyItem(volItem)
        volPlugin.setDisplayVisibility(volItem, 0)
        self.start_points.GetDisplayNode().SetVisibility(0)
        self.end_points.GetDisplayNode().SetVisibility(0)
    def test_summarize_Data_and_Create_Model(self):
        # Test summarize_Data_and_Create_Model function
        n_Entry, n_Target = self.logic.summarize_Data_and_Create_Model(self.start_points, self.end_points)
        if n_Entry != 48 or n_Target != 21:
            logging.warning("Nummber of entries and targets are not corrected")
        else:
            self.delayDisplay("Test passed: Model Creation")
    def test_Check_A_Point_Is_In_A_Label_Volume_Node(self):
        # Test Check_A_Point_Is_In_A_Label_Volume_Node function
        if True:
            logging.warning("Test failed: Check A Point Is In A Label Volume Node")
        else:
            self.delayDisplay("Test passed: Check A Point Is In A Label Volume Node")
    def test_Pathway_planning1(self):
        self.delayDisplay("Starting the test")

        # # Get/create input data

        # import SampleData

        # registerSampleData()
        # inputVolume = SampleData.downloadSample("Pathway_planning1")
        # self.delayDisplay("Loaded test data set")

        # inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(inputScalarRange[0], 0)
        # self.assertEqual(inputScalarRange[1], 695)

        # outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        # threshold = 100

        # # Test the module logic

        # logic = Pathway_planningLogic()

        # # Test algorithm with non-inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, True)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], threshold)

        # # Test algorithm with inverted threshold
        # logic.process(inputVolume, outputVolume, threshold, False)
        # outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        # self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        # self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")
        pass

    def object_Creation_For_Testing_Creating_Box_All(self,
                                                nodeNameEnd = "Standard",
                                                imageSize = [32, 32, 32],
                                                imageOrigin = [0.0, 0.0, 0.0],
                                                imageSpacing = [1.0, 1.0, 1.0],
                                                imageDirections = [[1,0,0], [0,1,0], [0,0,1]]):
        ###Create a volume
        nodeName = "Test_Volume_" + nodeNameEnd
        voxelType=vtk.VTK_UNSIGNED_CHAR
        fillVoxelValue = 0
        # Create an empty image volume, filled with fillVoxelValue
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Edit the data to create a square
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    imageData.SetScalarComponentFromFloat(i,j,k,0,1)
        # Create volume node
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        volumeNode.SetOrigin(imageOrigin)
        volumeNode.SetSpacing(imageSpacing)
        volumeNode.SetIJKToRASDirections(imageDirections)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(volumeNode), self.volumefortestfolderID)
        # Create a label map
        nodeName = "Test_Label_Volume_"+nodeNameEnd
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", nodeName)
        labelmapVolumeNode.SetOrigin(imageOrigin)
        labelmapVolumeNode.SetSpacing(imageSpacing)
        labelmapVolumeNode.SetIJKToRASDirections(imageDirections)
        labelmapVolumeNode.SetAndObserveImageData(imageData)
        labelmapVolumeNode.CreateDefaultDisplayNodes()
        labelmapVolumeNode.CreateDefaultStorageNode()
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(labelmapVolumeNode), self.labelvolumefortestfolderID)
        # Create a segmentation
        nodeName = "Test_Segmentation_"+nodeNameEnd
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(segmentationNode), self.segmentationfortestfolderID)
        # Create a model
        slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode, self.modelfortestfolderID)
        # Move to desired folder
        vtkIDlist = vtk.vtkIdList()
        self.shNode.GetItemChildren(self.modelfortestfolderID, vtkIDlist)
        self.shNode.SetItemName(vtkIDlist.GetId(vtkIDlist.GetNumberOfIds()-1), "Test_Model_"+nodeNameEnd)
        modelNode = self.shNode.GetItemDataNode(vtkIDlist.GetId(vtkIDlist.GetNumberOfIds()-1))
        # Move model folder to desired folder
        self.shNode.SetItemParent(self.modelfortestfolderID, self.shNode.GetSceneItemID())
        return volumeNode, labelmapVolumeNode, segmentationNode, modelNode
    def object_Creation_For_Testing_Creating_Box_Volume(self,
                                                nodeNameEnd = "Standard",
                                                imageSize = [32, 32, 32],
                                                imageOrigin = [0.0, 0.0, 0.0],
                                                imageSpacing = [1.0, 1.0, 1.0],
                                                imageDirections = [[1,0,0], [0,1,0], [0,0,1]]):
        ###Create a volume
        nodeName = "Test_Volume_" + nodeNameEnd
        voxelType=vtk.VTK_UNSIGNED_CHAR
        fillVoxelValue = 0
        # Create an empty image volume, filled with fillVoxelValue
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Edit the data to create a square
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    imageData.SetScalarComponentFromFloat(i,j,k,0,1)
        # Create volume node
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", nodeName)
        volumeNode.SetOrigin(imageOrigin)
        volumeNode.SetSpacing(imageSpacing)
        volumeNode.SetIJKToRASDirections(imageDirections)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(volumeNode), self.volumefortestfolderID)
        return volumeNode
    def object_Creation_For_Testing_Creating_Box_Label_Volume(self,
                                                nodeNameEnd = "Standard",
                                                imageSize = [32, 32, 32],
                                                imageOrigin = [0.0, 0.0, 0.0],
                                                imageSpacing = [1.0, 1.0, 1.0],
                                                imageDirections = [[1,0,0], [0,1,0], [0,0,1]]):
        ###Create a label volume
        voxelType=vtk.VTK_UNSIGNED_CHAR
        fillVoxelValue = 0
        # Create an empty image volume, filled with fillVoxelValue
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Edit the data to create a square
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    imageData.SetScalarComponentFromFloat(i,j,k,0,1)
        # Create a label map
        nodeName = "Test_Label_Volume_"+nodeNameEnd
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", nodeName)
        labelmapVolumeNode.SetOrigin(imageOrigin)
        labelmapVolumeNode.SetSpacing(imageSpacing)
        labelmapVolumeNode.SetIJKToRASDirections(imageDirections)
        labelmapVolumeNode.SetAndObserveImageData(imageData)
        labelmapVolumeNode.CreateDefaultDisplayNodes()
        labelmapVolumeNode.CreateDefaultStorageNode()
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(labelmapVolumeNode), self.labelvolumefortestfolderID)
        return labelmapVolumeNode
    def object_Creation_For_Testing_Creating_Box_Segmentation(self,
                                                nodeNameEnd = "Standard",
                                                imageSize = [32, 32, 32],
                                                imageOrigin = [0.0, 0.0, 0.0],
                                                imageSpacing = [1.0, 1.0, 1.0],
                                                imageDirections = [[1,0,0], [0,1,0], [0,0,1]]):
        ###Create a label volume
        voxelType=vtk.VTK_UNSIGNED_CHAR
        fillVoxelValue = 0
        # Create an empty image volume, filled with fillVoxelValue
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Edit the data to create a square
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    imageData.SetScalarComponentFromFloat(i,j,k,0,1)
        # Create a label map
        nodeName = "Test_Label_Volume_"+nodeNameEnd
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", nodeName)
        labelmapVolumeNode.SetOrigin(imageOrigin)
        labelmapVolumeNode.SetSpacing(imageSpacing)
        labelmapVolumeNode.SetIJKToRASDirections(imageDirections)
        labelmapVolumeNode.SetAndObserveImageData(imageData)
        labelmapVolumeNode.CreateDefaultDisplayNodes()
        labelmapVolumeNode.CreateDefaultStorageNode()
        # Create a segmentation
        nodeName = "Test_Segmentation_"+nodeNameEnd
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
        # Move to desired folder
        self.shNode.SetItemParent(self.shNode.GetItemByDataNode(segmentationNode), self.segmentationfortestfolderID)
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        return segmentationNode
    def object_Creation_For_Testing_Creating_Box_Model(self,
                                                nodeNameEnd = "Standard",
                                                imageSize = [32, 32, 32],
                                                imageOrigin = [0.0, 0.0, 0.0],
                                                imageSpacing = [1.0, 1.0, 1.0],
                                                imageDirections = [[1,0,0], [0,1,0], [0,0,1]]):
        ###Create a label volume
        voxelType=vtk.VTK_UNSIGNED_CHAR
        fillVoxelValue = 0
        # Create an empty image volume, filled with fillVoxelValue
        imageData = vtk.vtkImageData()
        imageData.SetDimensions(imageSize)
        imageData.AllocateScalars(voxelType, 1)
        imageData.GetPointData().GetScalars().Fill(fillVoxelValue)
        # Edit the data to create a square
        for i in range(16):
            for j in range(16):
                for k in range(16):
                    imageData.SetScalarComponentFromFloat(i,j,k,0,1)
        # Create a label map
        nodeName = "Test_Label_Volume_"+nodeNameEnd
        labelmapVolumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode", nodeName)
        labelmapVolumeNode.SetOrigin(imageOrigin)
        labelmapVolumeNode.SetSpacing(imageSpacing)
        labelmapVolumeNode.SetIJKToRASDirections(imageDirections)
        labelmapVolumeNode.SetAndObserveImageData(imageData)
        labelmapVolumeNode.CreateDefaultDisplayNodes()
        labelmapVolumeNode.CreateDefaultStorageNode()
        # Create a segmentation and model
        nodeName = "Test_Segmentation_"+nodeNameEnd
        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode", nodeName)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(labelmapVolumeNode, segmentationNode)
        slicer.modules.segmentations.logic().ExportAllSegmentsToModels(segmentationNode, self.modelfortestfolderID)
        # self.shNode.SetItemParent(exportModelFolderItemId, self.shNode.GetSceneItemID())
        vtkIDlist = vtk.vtkIdList()
        self.shNode.GetItemChildren(self.modelfortestfolderID, vtkIDlist)
        self.shNode.SetItemName(vtkIDlist.GetId(vtkIDlist.GetNumberOfIds()-1), "Test_Model_"+nodeNameEnd)
        modelNode = self.shNode.GetItemDataNode(vtkIDlist.GetId(vtkIDlist.GetNumberOfIds()-1))
        # Move model folder to desired folder
        self.shNode.SetItemParent(self.modelfortestfolderID, self.shNode.GetSceneItemID())
        slicer.mrmlScene.RemoveNode(labelmapVolumeNode)
        slicer.mrmlScene.RemoveNode(segmentationNode)
        return modelNode

    def simple_Task_Create_Folder(self, folder_name):
        folder_ID = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), folder_name)
        self.shNode.SetItemExpanded(folder_ID, 0)
        return folder_ID

            
