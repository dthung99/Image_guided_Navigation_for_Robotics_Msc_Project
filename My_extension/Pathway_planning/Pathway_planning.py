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
import numpy as np

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
        """Summarize the data and create model, segmentation"""
        # Get number of entry and target point
        n_Entry = entryNode.GetNumberOfControlPoints()
        n_Target = targetNode.GetNumberOfControlPoints()
        print(f"Number of entries is: {n_Entry}")
        print(f"Number of targets is: {n_Target}")
        # Create segmentation and model from label map
        list_of_Label_nodes = slicer.util.getNodesByClass("vtkMRMLLabelMapVolumeNode")
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        exportModelFolderItemId = self.simple_Task_Create_Folder("Models")
        exportSegmentFolderItemId = self.simple_Task_Create_Folder("Segments")
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
    def check_A_Point_Is_In_A_Label_Volume_Node(self, input_point_array: np.ndarray((100,3)), label_volume_node: vtkMRMLLabelMapVolumeNode, strictly_inside_mode = True):
        """Get a list of point as np.array and a label volume node"""
        """Return a np.array(-1,) of boolean where True if a point is inside, and False if it is outside"""
        """If strictly_inside_mode is True, the function will return 1 only if 8 cubes around it is inside
        If strictly_inside_mode is False, the function will return 1 if at least 1 in 9 cubes around it is inside"""
        # Check input types
        if type(input_point_array) != np.ndarray:
            return "check_A_Point_Is_In_A_Label_Volume_Node failed: Input point need to be a np.ndarray"
        if type(label_volume_node) != vtkMRMLLabelMapVolumeNode:
            return "check_A_Point_Is_In_A_Label_Volume_Node failed: Input volume need to be a vtkMRMLLabelMapVolumeNode"
        result = np.empty((input_point_array.shape[0],)) # Declare result vector
        strictly_inside_mode = strictly_inside_mode*8
        label_volume_array = slicer.util.arrayFromVolume(label_volume_node) # Conver the node into np array
        # Loop through list of points, convert to IJK space, and check value of 8 cubes around it.
        spacing = np.array(label_volume_node.GetSpacing())
        rotation = vtk.vtkMatrix4x4()
        label_volume_node.GetIJKToRASDirectionMatrix(rotation)
        rotation = slicer.util.arrayFromVTKMatrix(rotation)[0:3,0:3]
        origin = np.array(label_volume_node.GetOrigin())
        for i, input_point in enumerate(input_point_array):
            x, y, z  = np.round(self.convert_RAS_to_IJK(input = input_point,
                                                        spacing = spacing, 
                                                        rotation = rotation,
                                                        origin = origin)).astype(int)
            matrix_surround_current_point = self.get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position(x, y, z, label_volume_array) #The value of the image at x y z coordinate and 8 cubes around
            if np.sum(matrix_surround_current_point)>strictly_inside_mode:
                result[i] = True
            else:
                result[i] = False
        return result

    def convert_IJK_to_RAS(self,
                           input: np.ndarray((3,)),
                           spacing: np.ndarray((3,)) = np.array([1,1,1]), 
                           rotation: np.ndarray((3,3)) = np.eye(3),
                           origin: np.ndarray((3,)) = np.array([0,0,0])) -> np.ndarray((3,)):
        """Order for transformation: spacing -> rotate -> translate"""
        input = np.array(input).reshape(3,)
        spacing = np.array(spacing).reshape(3,)
        rotation = np.array(rotation).reshape(3,3)
        origin = np.array(origin).reshape(3,)
        return np.matmul(rotation, (input*spacing).reshape(-1,1)).reshape(-1,) + origin
    def convert_RAS_to_IJK(self,
                           input: np.ndarray((3,)),
                           spacing: np.ndarray((3,)) = np.array([1,1,1]), 
                           rotation: np.ndarray((3,3)) = np.eye(3),
                           origin: np.ndarray((3,)) = np.array([0,0,0])) -> np.ndarray((3,)):
        """Backward transformation of convert_IJK_to_RAS
        The input rotation matrix must not be singular"""
        input = np.array(input).reshape(3,)
        spacing = np.array(spacing).reshape(3,)
        rotation = np.array(rotation).reshape(3,3)
        origin = np.array(origin).reshape(3,)
        try:
            return np.matmul(np.linalg.inv(rotation), (input - origin).reshape(-1,1)).reshape(-1,)/spacing
        except np.linalg.LinAlgError:
            print("Convert_RAS_to_IJK failed, the rotation matrix must be non-singular")
            return
    def get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position(self, x: int, y: int, z: int, array: np.ndarray):
        result = np.zeros((3,3,3))
        for i in [x-1,x,x+1]:
            for j in [y-1,y,y+1]:
                for k in [z-1,z,z+1]:
                    condition = i>=array.shape[0] + j>=array.shape[1] + k>=array.shape[2] + i<0 + j<0 + k<0
                    condition = (i>=array.shape[0]) + (j>=array.shape[1]) + (k>=array.shape[2]) + (i<0) + (j<0) + (k<0)                    
                    if condition == 0:
                        result[i-x+1,j-y+1,k-z+1] = array[i,j,k]
                    else:
                        result[i-x+1,j-y+1,k-z+1] = 0
        return result

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

    def simple_Task_Create_Folder(self, folder_name):
        """Create a folder and collapse it"""
        folder_ID = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), folder_name)
        # Collapse the folder
        self.shNode.SetItemExpanded(folder_ID, 0)
        # Hide the folder
        pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
        volPlugin = pluginHandler.getOwnerPluginForSubjectHierarchyItem(folder_ID)
        volPlugin.setDisplayVisibility(folder_ID, 0)
        return folder_ID

# Pathway_planningTest
class Pathway_planningTest(ScriptedLoadableModuleTest):
    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()
        # Declare variable to control system
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.shNode.RemoveAllItems()
        self.logic = Pathway_planningLogic()
        self.test_data_folder_path = "D:\\Code\\3D_slicer\\Navigation_robotics\\Data\\Week_2_4_TestSet_Path_planning\\" #Path to place that store data
        # Create my custom logger
        logging_level = logging.INFO #Choose your logging level
        self.logger_custom = logging.getLogger("My_logger")
        self.logger_custom.handlers = []
        self.logger_custom.filters = []
        self.logger_custom.setLevel(logging_level)
        # # Create a file handler to log messages to a file
        file_handler = logging.StreamHandler()
        file_handler.setLevel(logging_level)
        self.logger_custom.addHandler(file_handler)
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
        # Creating common test objects
        self.test_standard_volume, self.test_standard_label_volume, self.test_standard_segmentation, self.test_standard_model = self.object_Creation_For_Testing_Creating_Box_All(nodeNameEnd = "Standard",
                                                            imageSize = [32, 32, 32],
                                                            imageOrigin = [1.0, 1.0, 0.0],
                                                            imageSpacing = [2.0, 2.0, 2.0],
                                                            imageDirections = [[np.cos(np.pi/4),-np.sin(np.pi/4),0], [np.sin(np.pi/4),np.cos(np.pi/4),0], [0,0,1]])
        # self.object_Creation_For_Testing_Creating_Box_Volume("Volume")
        # self.object_Creation_For_Testing_Creating_Box_Label_Volume("Label")
        # self.object_Creation_For_Testing_Creating_Box_Segmentation("Segmentation")
        # self.object_Creation_For_Testing_Creating_Box_Model("Model")


        self.test_check_A_Point_Is_In_A_Label_Volume_Node()
        self.unit_test_convert_IJK_to_RAS()
        self.unit_test_convert_RAS_tos_IJK()
        self.unit_test_get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position()
        
    def load_Data(self):
        """Load data and set their visual"""
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
        """Test summarize_Data_and_Create_Model function"""
        successful_tested = 1
        n_Entry, n_Target = self.logic.summarize_Data_and_Create_Model(self.start_points, self.end_points)
        successful_tested = not(n_Entry != 48 or n_Target != 21)
        self.simple_Task_Logging_Test_Outcome("Model Creation", successful_tested)
    def test_check_A_Point_Is_In_A_Label_Volume_Node(self):
        """Test Check_A_Point_Is_In_A_Label_Volume_Node function"""
        successful_tested = 1
        ### Example 1
        # Create a label volume node and point
        label_volume_node = self.test_standard_label_volume
        input_point = np.array([[1,1,0], [0,16.5,16.5], [10,0,16]])
        expected_result = np.array([True, True, False])
        result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node, strictly_inside_mode=False)
        # Check
        if not(np.array_equal(expected_result, result)):
            successful_tested = 0
        ### Example 2 - test the borderline point
        # Create a label volume node and point
        label_volume_node = self.test_standard_label_volume
        input_point = np.array([[1,1,0], [0,16.5,16.5], [10,0,16]])
        expected_result = np.array([False, True, False])
        result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node, strictly_inside_mode=True)
        # Check
        if not(np.array_equal(expected_result, result)):
            successful_tested = 0            
        ### Example 3
        # Create a label volume node and point
        label_volume_node = self.test_standard_label_volume
        input_point = None
        expected_result = "check_A_Point_Is_In_A_Label_Volume_Node failed: Input point need to be a np.ndarray"
        result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node)
        # Check
        if expected_result != result:
            successful_tested = 0
        ### Example 4
        # Create a label volume node and point
        label_volume_node = None
        input_point = np.array([0,0,0])
        expected_result = "check_A_Point_Is_In_A_Label_Volume_Node failed: Input volume need to be a vtkMRMLLabelMapVolumeNode"
        result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node)
        # Check
        if expected_result != result:
            successful_tested = 0
        
        # Log the result
        self.simple_Task_Logging_Test_Outcome("Check a Point is in a Label Volume Node", successful_tested)

    # def test_check_A_Line_Pass_Through_A_Label_Volume_Node(self):
    #     """Test check_A_Line_Pass_Through_A_Label_Volume_Node function"""
    #     successful_tested = 1
    #     ### Example 1
    #     # Create a label volume node and point
    #     label_volume_node = self.test_standard_label_volume
    #     input_point = np.array([[1,1,0], [0,16.5,16.5], [10,0,16]])
    #     expected_result = np.array([True, True, False])
    #     result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node, strictly_inside_mode=False)
    #     # Check
    #     if not(np.array_equal(expected_result, result)):
    #         successful_tested = 0
    #     ### Example 2 - test the borderline point
    #     # Create a label volume node and point
    #     label_volume_node = self.test_standard_label_volume
    #     input_point = np.array([[1,1,0], [0,16.5,16.5], [10,0,16]])
    #     expected_result = np.array([False, True, False])
    #     result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node, strictly_inside_mode=True)
    #     # Check
    #     if not(np.array_equal(expected_result, result)):
    #         successful_tested = 0            
    #     ### Example 3
    #     # Create a label volume node and point
    #     label_volume_node = self.test_standard_label_volume
    #     input_point = None
    #     expected_result = "check_A_Point_Is_In_A_Label_Volume_Node failed: Input point need to be a np.ndarray"
    #     result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node)
    #     # Check
    #     if expected_result != result:
    #         successful_tested = 0
    #     ### Example 4
    #     # Create a label volume node and point
    #     label_volume_node = None
    #     input_point = np.array([0,0,0])
    #     expected_result = "check_A_Point_Is_In_A_Label_Volume_Node failed: Input volume need to be a vtkMRMLLabelMapVolumeNode"
    #     result = self.logic.check_A_Point_Is_In_A_Label_Volume_Node(input_point, label_volume_node)
    #     # Check
    #     if expected_result != result:
    #         successful_tested = 0
        
    #     # Log the result
    #     self.simple_Task_Logging_Test_Outcome("Check a Point is in a Label Volume Node", successful_tested)


    def unit_test_convert_IJK_to_RAS(self):
        """Test convert_IJK_to_RAS function"""
        successful_tested = 1
        # Example 1
        original_vector = np.array([0,0,0])
        expected_result_vector = np.array([0,0,0])
        result_vector = self.logic.convert_IJK_to_RAS(input = original_vector,
                                                      spacing = np.array([1,1,1]),
                                                      rotation = np.eye(3),
                                                      origin = np.array([0,0,0]))
        if np.linalg.norm(result_vector-expected_result_vector) > 0.001:
            successful_tested = 0
        # Example 2
        original_vector = np.array([1,0,0])
        expected_result_vector = np.array([0,10,0])
        result_vector = self.logic.convert_IJK_to_RAS(input = original_vector,
                                                      spacing = np.array([5,5,5]),
                                                      rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]]),
                                                      origin = np.array([0,5,0]))
        if np.linalg.norm(result_vector-expected_result_vector) > 0.001:
            successful_tested = 0
        self.simple_Task_Logging_Unit_Test_Outcome("Convert IJK to RAS", successful_tested)
    def unit_test_convert_RAS_tos_IJK(self):
        """Test convert_RAS_tos_IJK function"""
        successful_tested = 1
        # Example 1
        original_vector = np.array([0,0,0])
        expected_result_vector = np.array([0,0,0])
        result_vector = self.logic.convert_RAS_to_IJK(input = original_vector,
                                                      spacing = np.array([1,1,1]),
                                                      rotation = np.eye(3),
                                                      origin = np.array([0,0,0]))
        if np.linalg.norm(result_vector-expected_result_vector) > 0.001:
            successful_tested = 0
        # Example 2
        original_vector = np.array([0,10,0])
        expected_result_vector = np.array([1,0,0])
        result_vector = self.logic.convert_RAS_to_IJK(input = original_vector,
                                                      spacing = np.array([5,5,5]),
                                                      rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]]),
                                                      origin = np.array([0,5,0]))
        if np.linalg.norm(result_vector-expected_result_vector) > 0.001:
            successful_tested = 0
        self.simple_Task_Logging_Unit_Test_Outcome("Convert RAS to IJK", successful_tested)
    def unit_test_get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position(self):
        """Test get_9x9_Matrix_From_Bigger_Matrix_at_one_Position function"""
        successful_tested = 1
        # Example 1
        input_x = 0
        input_y = 0
        input_z = 0
        input_array = np.arange(0,27,1).reshape((3,3,3))
        result = self.logic.get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position(input_x, input_y, input_z, input_array)
        expected_result = np.array([[[ 0, 0, 0],
                                     [ 0, 0, 0],
                                     [ 0, 0, 0]],
                                    [[ 0, 0, 0],
                                     [ 0, 0,  1],
                                     [ 0, 3,  4]],
                                    [[ 0, 0, 0],
                                     [ 0, 9, 10],
                                     [ 0, 12, 13]]])
        if np.linalg.norm(result-expected_result) > 0.001:
            successful_tested = 0
        # Example 2
        input_x = 6
        input_y = 6
        input_z = 6
        result = self.logic.get_3x3x3_Matrix_From_Bigger_Matrix_at_one_Position(input_x, input_y, input_z, input_array)
        expected_result = np.zeros((3,3,3))
        if np.linalg.norm(result-expected_result) > 0.001:
            successful_tested = 0
        self.simple_Task_Logging_Unit_Test_Outcome("Get 9x9 Matrix at one Position from a bigger Matrix ", successful_tested)

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
        """Create object for testing"""
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
        """Create a folder and collapse it"""
        folder_ID = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), folder_name)
        # Collapse the folder
        self.shNode.SetItemExpanded(folder_ID, 0)
        # Hide the folder
        pluginHandler = slicer.qSlicerSubjectHierarchyPluginHandler().instance()
        volPlugin = pluginHandler.getOwnerPluginForSubjectHierarchyItem(folder_ID)
        volPlugin.setDisplayVisibility(folder_ID, 0)
        return folder_ID
    def simple_Task_Logging_Test_Outcome(self, test_name = "Standard", successful_tested = False):
        """Logging test results"""
        if successful_tested:
            self.logger_custom.info("Test passed: " + test_name)
            self.delayDisplay("Test passed: " + test_name)
        else:
            self.logger_custom.warning("Test failed: " + test_name)         
    def simple_Task_Logging_Unit_Test_Outcome(self, test_name = "Standard", successful_tested = False):
        """Logging unit test results"""
        if successful_tested:
            self.logger_custom.info("Unit test passed: " + test_name)
        else:
            self.logger_custom.warning("Unit test failed: " + test_name)



