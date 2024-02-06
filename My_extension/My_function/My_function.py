from typing import Annotated, Optional
import vtk

import slicer
import SimpleITK
import sitkUtils

from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode, vtkMRMLModelNode

import random
import time
import numpy as np

class My_function(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("My function")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "My_modules")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["John Doe (AnyWare Corp.)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""This is a modules of common used function created by Dang The Hung (King's College London)""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""This is a modules of common used function created by Dang The Hung (King's College London)""")

@parameterNodeWrapper
class My_functionParameterNode:
    """
    No parameters are needed
    """
    # inputVolume: vtkMRMLScalarVolumeNode

class My_functionWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
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
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/My_function.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
        # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
        # "setMRMLScene(vtkMRMLScene*)" slot.
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class. Logic implements all computations that should be possible to run
        # in batch mode, without a graphical user interface.
        self.logic = My_functionLogic()

        # Connections

        # These connections ensure that we update parameter node when scene is closed
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Buttons
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
        # if not self._parameterNode.inputVolume:
        #     firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
        #     if firstVolumeNode:
        #         self._parameterNode.inputVolume = firstVolumeNode

    def setParameterNode(self, inputParameterNode: Optional[My_functionParameterNode]) -> None:
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
        self.ui.applyButton.enabled = True

    def onApplyButton(self) -> None:
        """Run processing when user clicks "Apply" button."""
        with slicer.util.tryWithErrorDisplay(_("Failed to compute results."), waitCursor=True):
            # Compute output
            self.logic.process()

class My_functionLogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)

    def getParameterNode(self):
        return My_functionParameterNode(super().getParameterNode())

    def process(self) -> None:
        my_object = My_functionLogic()
        # Use the dir() function to get a list of functions in the class
        functions = [func for func in dir(my_object) if callable(getattr(my_object, func))]
        # Print the list of functions
        for func in functions:
            if not func.startswith("__"):
                print(func)

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
            print("The rotation matrix must be non-singular")
            return

    def find_Triangle_Height(self, x: np.ndarray((3,)), y: np.ndarray((3,)), z: np.ndarray((3,))) -> float:
        """Find the distance between x and yz"""
        return np.linalg.norm(np.cross((x-z),(x-y)))/np.linalg.norm(y-z)
    
    def check_Projection_in_middle(self, x: np.ndarray((3,)), y: np.ndarray((3,)), z: np.ndarray((3,)), error: float = 0.01) -> bool:
        """Check whether the projection of x is on yz"""
        height_square = self.find_Triangle_Height(x,y,z)**2
        comparison = np.sqrt(np.dot(x-y,x-y)-height_square) + np.sqrt(np.dot(x-z,x-z)-height_square) - np.linalg.norm(y-z)
        if comparison < error:
            return True
        else:
            return False

    def find_Intersection_Line_and_Model(self, start_point: np.ndarray((3,)), end_point: np.ndarray((3,)), model: vtkMRMLModelNode, distance_error: float = 0.1) -> np.ndarray((3,)):
        '''Find intersection of a line and a model by splitting the
        segment into halves until the distance is less than needed'''
        start_point = np.array(start_point)
        end_point = np.array(end_point)

        Model_formula = vtk.vtkImplicitPolyDataDistance()
        Model_formula.SetInput(model.GetPolyData())
        a = Model_formula.FunctionValue(start_point)
        b = Model_formula.FunctionValue(end_point)

        if a*b > 0:
            print("Please select 1 point inside and 1 point outside the model")
            return
        
        while abs(a) > distance_error:
            middle_point = (start_point + end_point)/2
            c = Model_formula.FunctionValue(middle_point)
            if a*c > 0:
                start_point = middle_point
                a = c
            else:
                end_point = middle_point
                b=c
        return start_point

    def find_nearest_Cell(self, point: np.ndarray((3,)), model: vtkMRMLModelNode) -> int:
        """Find the ID of nearest cell from a model
        Reference: https://slicer.readthedocs.io/en/latest/developer_guide/script_repository.html#select-cells-of-a-model-using-markups-point-list"""
        point = np.array(point)
        cell_locator = vtk.vtkCellLocator()
        cell_locator.SetDataSet(model.GetMesh())
        cell_locator.BuildLocator()
        closestPoint = [0.0, 0.0, 0.0]
        cellObj = vtk.vtkGenericCell()
        cellId = vtk.mutable(0)
        subId = vtk.mutable(0)
        dist = vtk.mutable(0.0)
        cell_locator.FindClosestPoint(point, closestPoint, cellObj, cellId, subId, dist)
        return int(cellId)

    def find_Unit_Normal_Vector_of_Cell_from_model(self, cell_ID: int, model: vtkMRMLModelNode) -> np.ndarray((3,)):
        """Find the normal vector at certain cell ID of a model"""
        mesh_model = model.GetMesh()
        cell = mesh_model.GetCell(cell_ID)
        vertex_1 = np.array(mesh_model.GetPoint(cell.GetPointId(0)))
        vertex_2 = np.array(mesh_model.GetPoint(cell.GetPointId(1)))
        vertex_3 = np.array(mesh_model.GetPoint(cell.GetPointId(2)))
        cross = np.cross(vertex_1-vertex_2, vertex_1-vertex_3)
        return cross/np.linalg.norm(cross)


class My_function_for_Projects(ScriptedLoadableModuleLogic):
    """The functions here designed for specific task for my project"""
    def __init__(self) -> None:
        """Called when the logic class is instantiated. Can be used for initializing member variables."""
        ScriptedLoadableModuleLogic.__init__(self)
        self.randomlist = []
        self.check_Segment_Violation_Bad_Pairs = []
        self.check_Segment_Violation_Good_Pairs = []
        self.find_Pathway_Bad_Pairs = []
        self.find_Pathway_Good_Pairs = []
        self.shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
        self.exportFolderItemId = self.shNode.GetItemByName("qualified_Lines")
        if self.exportFolderItemId == 0:
            self.exportFolderItemId = self.shNode.CreateFolderItem(self.shNode.GetSceneItemID(), "qualified_Lines")
        self.logic = My_functionLogic()
        self.optimized_start_point = None
        self.optimized_end_point = None
        self.global_min_distance_to_obstacle = 0

    def measure_implement_time_1(self, mesh) -> float:        
        start_time = time.time()
        label_function = vtk.vtkImplicitPolyDataDistance()
        label_function.SetInput(mesh.GetPolyData())
        label_function.EvaluateFunction([0,0,0])
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        return

    def measure_implement_time_2(self, volumn_node) -> float:
        start_time = time.time()
        distanceFilter = SimpleITK.DanielssonDistanceMapImageFilter()
        distance_map = distanceFilter.Execute(sitkUtils.PullVolumeFromSlicer(volumn_node))
        sitkUtils.PushVolumeToSlicer(distance_map)
        end_time = time.time()
        execution_time = end_time - start_time
        print("Execution time:", execution_time, "seconds")
        return

    def start_randomlist(self, a, b = 1, c = 1) -> None:
        self.randomlist = []
        for i in range(c):
            for j in range(b):
                for k in range(a):
                    self.randomlist.append((k, j, i))
        self.random_out = []
        return
    
    def check_Segment_Violation_set_Target(self, *targets):
        # targets is vtkMRMLModelNode
        self.targets = np.empty((0,3))
        for target in targets:
            self.targets = np.append(self.targets, slicer.util.arrayFromModelPoints(target), axis=0)
        return

    def check_Segment_Violation_set_Obstacle(self, *obstacles):
        # obtacles is vtkMRMLModelNode
        self.obstacles = np.empty((0,3))
        for obstacle in obstacles:
            self.obstacles = np.append(self.obstacles, slicer.util.arrayFromModelPoints(obstacle), axis=0)
        return

    def check_Segment_Violation(self, start_point, end_point, spacing) -> bool:
        """Check if the tool violate the constraints"""
        # start_point and end_point are vtkMRMLMarkupsFiducialNode
        # model is vtkMRMLModelNode
        random_item = random.choice(self.randomlist)
        self.randomlist.remove(random_item)
        self.random_out.append(random_item)
        start_point = slicer.util.arrayFromMarkupsControlPoints(start_point)[random_item[0]]
        end_point = slicer.util.arrayFromMarkupsControlPoints(end_point)[random_item[1]]
        models = self.obstacles

        # for start_point in start_points:
        #     for end_point in end_points:
        #         for model in models:
        #             print("2")
        # Check violation in IJK coordinate
        violation = False
        for model in models:
            in_middle = self.logic.check_Projection_in_middle(model, start_point, end_point)
            distance = self.logic.find_Triangle_Height(model, start_point, end_point)
            if in_middle and distance < spacing:
                violation = True
        if not violation:
            shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
            exportFolderItemId = shNode.GetItemByName("qualified_Lines")
            if exportFolderItemId == 0:
                exportFolderItemId = shNode.CreateFolderItem(shNode.GetSceneItemID(), "qualified_Lines")
            qualified_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            qualified_line.SetName("qualified_Line")            
            qualified_line.SetLineStartPosition(start_point)
            qualified_line.SetLineEndPosition(end_point)
            qualified_line.GetDisplayNode().PropertiesLabelVisibilityOff()
            shNode.SetItemParent(shNode.GetItemByDataNode(qualified_line), exportFolderItemId)

            self.check_Segment_Violation_Good_Pairs.append(random_item)            
            print("Good to go")
        else:
            self.check_Segment_Violation_Bad_Pairs.append(random_item)
            print("Violate!!!")
        return

    def find_Pathway_set_Target(self, *targets):
        # targets is vtkMRMLModelNode
        self.find_Pathway_targets = []
        for target in targets:
            self.find_Pathway_targets.append(target)
        return

    def find_Pathway_set_Obstacle(self, *obstacles):
        # obtacles is vtkMRMLModelNode
        adding_tool = vtk.vtkAppendPolyData()
        for obstacle in obstacles:
            adding_tool.AddInputData(obstacle.GetPolyData())        
        adding_tool.Update()
        new_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ObstacleModel")
        new_model.SetAndObservePolyData(adding_tool.GetOutput())
        self.find_Pathway_obstacles = [new_model]
        return

    def find_Pathway_set_Model_for_Angle_Require(self, *models):
        # model is vtkMRMLModelNode
        self.find_Pathway_Model_for_Angle_Require = []
        for model in models:
            self.find_Pathway_Model_for_Angle_Require.append(model)
        return

    def find_Pathway_Check_Angle_Require(self, start_point, end_point, angle_versus_normal_vector:float=np.pi*55/180) -> bool:
        for model in self.find_Pathway_Model_for_Angle_Require:
            Model_formula = vtk.vtkImplicitPolyDataDistance()
            Model_formula.SetInput(model.GetPolyData())
            a = Model_formula.FunctionValue(start_point)
            b = Model_formula.FunctionValue(end_point)
            if a*b > 0:
                return False
            contact_point = self.logic.find_Intersection_Line_and_Model(start_point, end_point, model)
            cell_ID = self.logic.find_nearest_Cell(contact_point, model)
            normal_vector = self.logic.find_Unit_Normal_Vector_of_Cell_from_model(cell_ID, model)
            # Find the angle of two vector
            cos_value = np.dot(normal_vector, end_point - start_point)/(np.linalg.norm(normal_vector)*np.linalg.norm(end_point - start_point))
            self.find_Pathway_Angle = np.arccos(abs(cos_value))
            if self.find_Pathway_Angle > angle_versus_normal_vector:              
                return False
        return True

    def find_Pathway(self, start_point, end_point, discretize_distance) -> None:
        """Check if the tool hit obstacle and get to target"""
        # start_point and end_point are vtkMRMLMarkupsFiducialNode
        # model is vtkMRMLModelNode
        random_item = random.choice(self.randomlist)
        self.randomlist.remove(random_item)
        self.random_out.append(random_item)
        
        start_point = slicer.util.arrayFromMarkupsControlPoints(start_point)[random_item[0]]
        end_points = slicer.util.arrayFromMarkupsControlPoints(end_point)
        Model_formula_target = vtk.vtkImplicitPolyDataDistance()
        Model_formula_obstacle = vtk.vtkImplicitPolyDataDistance()
        Model_formula_target.SetInput(self.find_Pathway_targets[0].GetPolyData())
        for end_point in end_points:      
            # Check the targets
            if Model_formula_target.FunctionValue(end_point) <= 0:
                continue

            # Check the obstacles
            unit_vector = (end_point - start_point)/np.linalg.norm(end_point - start_point)*discretize_distance
            number_of_segments = int(np.linalg.norm(end_point - start_point)//discretize_distance)

            Model_formula_obstacle.SetInput(self.find_Pathway_obstacles[0].GetPolyData())
            discretize_vector = start_point
            i = 0
            hitting = False
            if Model_formula_obstacle.FunctionValue(end_point) > -discretize_distance/2:
                hitting = True
            # Discretize and check for hitting
            while hitting == False and i < (number_of_segments + 1):
                if Model_formula_obstacle.FunctionValue(discretize_vector) > -discretize_distance/2:
                    hitting = True
                discretize_vector = discretize_vector + unit_vector
                i += 1
            if hitting == True:
                continue
            # Check for angle between the line and model
            if not self.find_Pathway_Check_Angle_Require(start_point, end_point):
                continue
            qualified_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            qualified_line.SetName(f"qualified_Line_angle_{np.round(self.find_Pathway_Angle*180/np.pi,1)}_degree")            
            qualified_line.SetLineStartPosition(start_point)
            qualified_line.SetLineEndPosition(end_point)
            qualified_line.GetDisplayNode().PropertiesLabelVisibilityOff()            
            self.shNode.SetItemParent(self.shNode.GetItemByDataNode(qualified_line), self.exportFolderItemId)
            qualified_line.LockedOn()
            self.find_Pathway_Good_Pairs.append(random_item)            
            # print(Model_formula_obstacle.FunctionValue(end_point))
            print("Good to go")
            return
        return
    
    def find_Pathway_Better_Set_Target(self, end_point) -> None:
        """Check if the tool get to target and removed unnecessary target"""
        # end_point are vtkMRMLMarkupsFiducialNode
        self.end_points = slicer.util.arrayFromMarkupsControlPoints(end_point)
        Model_formula_target = vtk.vtkImplicitPolyDataDistance()
        Model_formula_target.SetInput(self.find_Pathway_targets[0].GetPolyData())
        i = 0
        while i < len(self.end_points):
            if Model_formula_target.FunctionValue(self.end_points[i]) <= 0:
                self.end_points = np.delete(self.end_points, i, axis=0)
            else:
                i+=1
        return

    def find_Pathway_set_Obstacle_to_maximize_distance(self, *obstacles):
        # obtacles is vtkMRMLModelNode
        adding_tool = vtk.vtkAppendPolyData()
        for obstacle in obstacles:
            adding_tool.AddInputData(obstacle.GetPolyData())        
        adding_tool.Update()
        new_model = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode", "ObstacleModel_to_check_distance")
        new_model.SetAndObservePolyData(adding_tool.GetOutput())
        self.find_Pathway_obstacles_to_maximize_distance = [new_model]
        return

    def find_Pathway_Better(self, start_point, discretize_distance) -> None:
        """Check if the tool hit obstacle and get to target"""
        random_item = random.choice(self.randomlist)
        self.randomlist.remove(random_item)
        self.random_out.append(random_item)

        start_point = slicer.util.arrayFromMarkupsControlPoints(start_point)[random_item[0]]
        end_points = self.end_points
        Model_formula_target = vtk.vtkImplicitPolyDataDistance()
        Model_formula_obstacle = vtk.vtkImplicitPolyDataDistance()
        Model_formula_obstacle_to_maximize_distance = vtk.vtkImplicitPolyDataDistance()
        Model_formula_target.SetInput(self.find_Pathway_targets[0].GetPolyData())
        min_distance = 10000
        for end_point in end_points:
            # Check the obstacles
            unit_vector = (end_point - start_point)/np.linalg.norm(end_point - start_point)*discretize_distance
            number_of_segments = int(np.linalg.norm(end_point - start_point)//discretize_distance)

            Model_formula_obstacle.SetInput(self.find_Pathway_obstacles[0].GetPolyData())
            discretize_vector = start_point
            i = 0
            hitting = False
            if Model_formula_obstacle.FunctionValue(end_point) > -discretize_distance/2:
                hitting = True
            # Discretize and check for hitting
            while hitting == False and i < (number_of_segments + 1):
                discretize_distance = Model_formula_obstacle.FunctionValue(discretize_vector)
                if discretize_distance > -discretize_distance/2:
                    hitting = True
                elif -discretize_distance < min_distance:
                    min_distance = -discretize_distance
                discretize_vector = discretize_vector + unit_vector
                i += 1
            if hitting == True:
                continue
            # Check for angle between the line and model
            if not self.find_Pathway_Check_Angle_Require(start_point, end_point):
                continue
            # Find minimum distance to obstacle
            Model_formula_obstacle_to_maximize_distance.SetInput(self.find_Pathway_obstacles_to_maximize_distance[0].GetPolyData())
            discretize_vector = start_point
            i = 0
            while i < (number_of_segments + 1):
                discretize_distance = Model_formula_obstacle_to_maximize_distance.FunctionValue(discretize_vector)
                if -discretize_distance < min_distance:
                    min_distance = -discretize_distance
                discretize_vector = discretize_vector + unit_vector
                i += 1
            if min_distance > self.global_min_distance_to_obstacle:
                self.optimized_start_point = start_point
                self.optimized_end_point =  end_point
                self.optimized_Angle = self.find_Pathway_Angle
                self.global_min_distance_to_obstacle = min_distance
            # qualified_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
            # qualified_line.SetName(f"qualified_Line_angle_{np.round(self.find_Pathway_Angle*180/np.pi,1)}_degree_distance: {min_distance} mm")            
            # qualified_line.SetLineStartPosition(start_point)
            # qualified_line.SetLineEndPosition(end_point)
            # qualified_line.GetDisplayNode().PropertiesLabelVisibilityOff()            
            # self.shNode.SetItemParent(self.shNode.GetItemByDataNode(qualified_line), self.exportFolderItemId)
            # qualified_line.LockedOn()
            self.find_Pathway_Good_Pairs.append(random_item)            
            # print(Model_formula_obstacle.FunctionValue(end_point))
            print("Good to go")
            return
        return

    def show_Optimized_line(self):
        qualified_line = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsLineNode")
        qualified_line.SetName(f"Optimized_Line__angle_{np.round(self.optimized_Angle*180/np.pi,1)}_degree_distance: {self.global_min_distance_to_obstacle} mm")            
        qualified_line.SetLineStartPosition(self.optimized_start_point)
        qualified_line.SetLineEndPosition(self.optimized_end_point)
        qualified_line.GetDisplayNode().PropertiesLabelVisibilityOff()            
        qualified_line.LockedOn()

    def find_Pathway_set_one_distance_Map(self, distance_label_maps) -> np.ndarray:
        # obtacles is vtkMRMLModelNode
        distanceFilter = SimpleITK.DanielssonDistanceMapImageFilter()
        distanceFilter.SetSquaredDistance(True)
        distanceFilter.UseImageSpacingOn()
        distance_map = distanceFilter.Execute(sitkUtils.PullVolumeFromSlicer(distance_label_maps))
        output_array = SimpleITK.GetArrayFromImage(distance_map)
        return output_array

