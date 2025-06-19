"""
modules.py
This module provides classes and utilities for unit conversion, logging, object serialization, directory management, variable storage, mesh parameter definition, and the generation of GATOR panel geometry and mesh using GMSH.
Classes:
--------
Units:
    Provides static methods for converting between meters and millimeters, and between degrees and radians.
Utils:
    Contains static utility methods including:
        - logger: Decorator for logging function execution.
        - save_object: Save Python objects using pickle or JSON.
        - load_object: Load Python objects using pickle or JSON.
        - format_repr: Format object representations for display.
        - create_case_dir: Create structured directories for simulation cases.
Variables:
    General-purpose class for storing and representing named variables as attributes.
Mesh_params:
    Represents mesh parameters for finite element models, including element type and mesh density.
GATOR:
    Class for defining and generating the geometry and mesh of a GATOR panel using GMSH.
    - Initializes with variables and mesh parameters.
    - Computes derived geometric variables.
    - Generates mesh and geometry, exporting to STEP and ABAQUS INP files.
Usage:
------
- Use the `Units` class for unit conversions.
- Use the `Utils` class for logging, object serialization, and directory management.
- Use the `variables` class to store arbitrary named variables.
- Use the `meshparams` class to define mesh settings.
- Use the `GATOR` class to generate panel geometry and mesh.
Dependencies:
-------------
- numpy
- gmsh
- os
- pickle
- json
- time
- traceback
- threading
Example:
--------
"""

import os

import pickle
import json

import time
import traceback
import threading

from typing import Any, Callable

import numpy as np
import gmsh # ignore type

class Utils:
    """
    A utility class providing methods used for miscellaneous tasks.
    """

    @staticmethod
    def logger(function: Callable, path: str = "logger"):
        """
        A decorator that logs the start and end time of a function execution, along with its arguments.
        It writes the function name, arguments, and the time taken to execute the function to a log file.
        """

        log_file = f"{path}.txt"

        def wrapper(*args, **kwargs):
            before = time.time()
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(
                    f"{time.strftime('%d/%m/%Y %H:%M')} :: Calling {function.__name__} with arguments: {args}, {kwargs}\n"
                )
            stop_progress = threading.Event()

            def show_progress():
                while not stop_progress.is_set():
                    elapsed = time.time() - before
                    duration = time.strftime("%H:%M:%S", time.gmtime(elapsed))
                    print(
                        f"\rRunning {function.__name__}... elapsed: {duration}",
                        end="",
                        flush=True,
                    )
                    stop_progress.wait(1)

            progress_thread = threading.Thread(target=show_progress)
            progress_thread.start()

            try:
                value = function(*args, **kwargs)
                after = time.time()
                duration = time.strftime("%H:%M:%S", time.gmtime(after - before))
                stop_progress.set()
                progress_thread.join()
                print(
                    f"\rCompleted {function.__name__} in {duration} (hr:min:s)           "
                )
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(
                        f"{time.strftime('%d/%m/%Y %H:%M')} :: Completed {function.__name__} in {duration} (hr:min:s)\n"
                    )
                return value
            except (OSError, ValueError, RuntimeError) as e:
                after = time.time()
                duration = time.strftime("%H:%M:%S", time.gmtime(after - before))
                stop_progress.set()
                progress_thread.join()
                print(
                    f"\rFailed {function.__name__} after {duration} (hr:min:s)           "
                )
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(
                        f"{time.strftime('%d/%m/%Y %H:%M')} :: Failed {function.__name__} after {duration} (hr:min:s) due to the error: {e}\n"
                    )
                    log.write(traceback.format_exc())
                    print(traceback.format_exc())
                return None

        return wrapper

    @staticmethod
    def save_object(obj:Any, path:str, method:str):
        """
        Saves a Python object to a file using the specified serialization method.

        Parameters:
            obj (any): The Python object to be saved.
            path (str): The file path (without extension) where the object will be saved.
            method (str): The serialization method to use. Supported values are:
                - 'pickle': Saves the object using Python's pickle module (binary format).
                - 'json': Saves the object in JSON format (text format).
        """
        match method:
            case "pickle":
                try:
                    with open(path + ".pickle", "wb") as f:
                        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
                except (pickle.PickleError, OSError) as ex:
                    print("ERROR: pickle save failed: ", ex)
            case "json":
                try:
                    with open(path + ".json", "w", encoding="utf-8") as f:
                        json.dump(obj, f, sort_keys=True, indent=4)
                except (TypeError, OSError, json.JSONDecodeError) as ex:
                    print("ERROR: json save failed: ", ex)
            case _:
                print("ERROR: Wrong fail saving method")

    @staticmethod
    def load_object(path:str, method:str) -> Any:
        """
        Loads a Python object from a file using the specified serialization method.

        Parameters:
            path (str): The base file path (without extension) from which to load the object.
            method (str): The serialization method to use ('pickle' or 'json').

        Returns:
            obj: The loaded Python object, or None if loading fails.
        """

        obj = None  # Ensure obj is always defined
        match method:
            case "pickle":
                try:
                    with open(path + ".pickle", "rb") as f:
                        obj = pickle.load(f, encoding="latin1")
                except (pickle.PickleError, OSError) as ex:
                    print("ERROR: pickle load failed: ", ex)
            case "json":
                try:
                    with open(path + ".json", "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except (TypeError, OSError, json.JSONDecodeError) as ex:
                    print("ERROR: json load failed: ", ex)
            case _:
                print("ERROR: Wrong fail saving method")
        return obj

    @staticmethod
    def format_repr(fields: dict | list | tuple | np.ndarray , indent: int = 0) -> str:
        """
        Recursively formats a dictionary or list as a tree-like string for display.
        """

        lines = []
        prefix = " " * (indent * 2)
        if isinstance(fields, dict):
            for key, value in fields.items():
                if isinstance(value, (int, float, str, bool)):
                    lines.append(f"{prefix}{key} : {value}")
                elif isinstance(value, (list, tuple, np.ndarray)) and isinstance(value[0], (int, float, str, bool)):
                    lines.append(f"{prefix}{key} : {value}")
                else:
                    lines.append(f"{prefix}{key} : ")
                    lines.append(Utils.format_repr(value, indent + 1))
        elif isinstance(fields, (list, tuple, np.ndarray)):
            if isinstance(fields[0], (int, float, str, bool)):
                lines.append(f"{prefix}{fields}")
            else:
                for item in fields:
                    lines.append(Utils.format_repr(item, indent + 1))            
        else:
            try:
                lines.append(Utils.format_repr(fields.__dict__, indent + 1))
            except Exception:
                raise TypeError(f"Error: Undefined variable type:{type(fields)}")
        return "\n".join(lines)

    @staticmethod
    def create_case_dir(case_name:str) -> dict:
        """
        Creates a structured directory tree for a simulation case.

        Parameters:
            case_name (str): Name of the case directory to create.

        Returns:
            dict: Paths to the main directories.
        """
        base_dir = os.path.abspath(os.getcwd())
        run_dir = os.path.join(base_dir, "run_dir")
        case_dir = os.path.join(base_dir, case_name)

        # Main directories
        dirs = {"CODE_DIR": base_dir, "RUN_DIR": run_dir, "CASE_DIR": case_dir}

        for path in dirs.values():
            os.makedirs(path, exist_ok=True)

        # Subfolders within the case directory
        subfolders = [
            "mesh",
            "input",
            "inp",
            "msg",
            "dat",
            "odb",
            "report",
            "data",
            "fig",
            "log",
            "sta",
            "temp",
        ]
        for sub in subfolders:
            os.makedirs(os.path.join(case_dir, sub), exist_ok=True)

        # Structured data folders
        for sub in ["inp", "data"]:
            os.makedirs(os.path.join(case_dir, sub, "serialised"), exist_ok=True)

        return dirs

class Units:
    """
    A class providing unit confversion methods.
    """

    @staticmethod
    def m2mm(value: float) -> float:
        """
        Converts a value from meters to millimeters.

        Parameters:
            value (float): The value in meters to be converted.

        Returns:
            float: The equivalent value in millimeters.
        """
        return value * 1000.0

    @staticmethod
    def mm2m(value: float) -> float:
        """
        Converts a value from millimeters to meters.

        Parameters:
            value (float): The value in millimeters to be converted.

        Returns:
            float: The converted value in meters.
        """
        return value / 1000.0

    @staticmethod
    def deg2rad(value: float) -> float:
        """
        Converts an angle from degrees to radians.

        Parameters:
            value (float): Angle in degrees.

        Returns:
            float: Angle in radians.
        """
        return np.deg2rad(value)

    @staticmethod
    def rad2deg(value: float) -> float:
        """
        Converts an angle from radians to degrees.

        Parameters:
            value (float): Angle in radians.

        Returns:
            float: Angle converted to degrees.
        """
        return np.rad2deg(value)

class PanelVariables:
    """
    A class defining GATOR panel geometry variables.
    """

    def __init__(
        self,
        core_type : str = "ZPR",
        mql: list = [True, True, True],
        nl: list = [1, 1],
        theta: float = np.deg2rad(60.0),
        chevron_wall_length: float = Units.mm2m(12.5),
        chevron_thickness: float = Units.mm2m(1.0),
        chevron_separation: float = Units.mm2m(6.0),
        rib_thickness: float = Units.mm2m(1.0),
        core_thickness: float = Units.mm2m(11.0),
        facesheet_thickness: float = Units.mm2m(0.8)
    ) -> None:
        """
        Initializes the variables object with default parameters for a GATOR panel.

        Parameters:
            mql (list): Bool for each axis for mirroring repeating unit to form unit cell.
            nl (list): Number of unit cell along each in-plane axis.
            theta (float): Chevron angle (in radians).
            chevron_wall_length (float): Length of the chevron wall (in meters).
            chevron_thickness (float): Thickness of the chevron wall (in meters).
            chevron_separation (float): Separation between chevrons (in meters).
            rib_thickness (float): Thickness of the rib (in meters).
            core_thickness (float): Thickness of the core (in meters).
            facesheet_thickness (float): Thickness of the facesheet (in meters).
        """
        self.core_type : str = core_type
        self.mql: list = mql
        self.nl: list = nl
        self.theta: float = theta
        self.chevron_wall_length: float = chevron_wall_length
        self.chevron_thickness: float = chevron_thickness
        self.chevron_separation: float = chevron_separation
        self.rib_thickness: float = rib_thickness
        self.core_thickness: float = core_thickness
        self.facesheet_thickness: float = facesheet_thickness

    def __repr__(self):
        return Utils.format_repr(self.__dict__)
    
class DerivedPanelVariables:

    def __init__(
        self, 
        variables : PanelVariables
    ) -> None:
        """
        Initializes the class with the given panel variables.
        Args:
            variables (PanelVariables): An instance containing the panel's configuration variables.
        """

        self.variables = variables

        if self.variables.core_type == "ZPR":
            self.GatorPanel()

    def GatorPanel(self):
        """
        Initializes and computes geometric and connectivity properties for a GatorPanel structure.
        This method calculates key geometric parameters and sets up node coordinates, line connectivity, and surface connectivity for the GatorPanel based on the current values in `self.variables`. The computed attributes are used for downstream structural analysis and modeling.
        Attributes Set:
            si (float): Internal chevron spacing.
            so (float): External chevron spacing.
            qlx (float): Length of repeating unit in the x-direction.
            qly (float): Length of repeating unit in the y-direction.
            qlz (float): Length of repeating unit facesheet thickness.
            core_area_ratio (float): Ratio of core area to total rectangular area.
            coordinates (dict): Dictionary mapping node labels to 3D coordinates (as numpy arrays).
            line_connectivity (list): List of line connectivity definitions between nodes.
            solid_surface_connectivity (list): List defining solid surface connectivity and orientation.
            facesheet_surfaces_connectivity (list): List defining facesheet surface connectivity.
            core_end_surface_box (list): List defining the core end surface box connectivity.
        Notes:
            - All geometric calculations use the current values in `self.variables`.
        """

        self.si: float = 0.5 * self.variables.chevron_thickness / np.cos(self.variables.theta) + 0.5 * self.variables.chevron_separation
        self.so: float = (
            +0.5 * self.variables.chevron_separation
            + 0.5 * self.variables.chevron_thickness / np.cos(self.variables.theta)
            - 0.5 * self.variables.rib_thickness * np.tan(self.variables.theta)
        )
        self.qlx: float = self.variables.chevron_wall_length * np.cos(self.variables.theta)
        self.qly: float = self.si + self.so + self.variables.chevron_wall_length * np.sin(self.variables.theta)
        self.qlz: float = self.variables.core_thickness / 2.0 + self.variables.facesheet_thickness

        self.core_area_ratio: float = (
            +self.qly * self.variables.rib_thickness / 2.0  # rib area
            + (self.variables.chevron_wall_length - (self.variables.rib_thickness / 2.0) / np.cos(self.variables.theta))
            * self.variables.chevron_thickness  # chevron area
        ) / (
            self.qlx * self.qly  # total rectangular area
        )

        # Define node coordinates using lists (not numpy arrays)
        self.coordinates: dict = {
            "1": [-self.qlx, 0.0, -self.qlz],
            "2": [
                -self.qlx,
                -self.qly
                + self.so
                + self.variables.rib_thickness / 2 * np.tan(self.variables.theta)
                + self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "3": [
                -self.qlx,
                -self.qly
                + self.so
                + self.variables.rib_thickness / 2 * np.tan(self.variables.theta)
                - self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "4": [-self.qlx, -self.qly, -self.qlz],
            "5": [-self.qlx + self.variables.rib_thickness / 2, 0.0, -self.qlz],
            "6": [
                -self.qlx + self.variables.rib_thickness / 2,
                -self.qly
                + self.so
                + self.variables.rib_thickness / 2 * np.tan(self.variables.theta)
                + self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "7": [
                -self.qlx + self.variables.rib_thickness / 2,
                -self.qly
                + self.so
                + self.variables.rib_thickness / 2 * np.tan(self.variables.theta)
                - self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "8": [-self.qlx + self.variables.rib_thickness / 2, -self.qly, -self.qlz],
            "9": [0.0, 0.0, -self.qlz],
            "10": [
                0.0,
                -self.si + self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "11": [
                0.0,
                -self.si - self.variables.chevron_thickness / 2 / np.cos(self.variables.theta),
                -self.qlz,
            ],
            "12": [0.0, -self.qly, -self.qlz],
        }
        # Convert all coordinate lists to numpy arrays for downstream compatibility
        for key in self.coordinates:
            self.coordinates[key] = np.array(self.coordinates[key], dtype=float)

        # Define line connectivity
        self.line_connectivity : list = [
            [[1, 5], [2, 6], [3, 7], [4, 8]],  # horizontal, left to right, column 1
            [
                [5, 9],
                [6, 10],
                [7, 11],
                [8, 12],
            ],  # horizontal, left to right, column 2
            [[1, 2], [5, 6], [9, 10]],  # vertical, top to bottom, row 1
            [[2, 3], [6, 7], [10, 11]],  # vertical, top to bottom, row 2
            [[3, 4], [7, 8], [11, 12]],  # vertical, top to bottom, row 3
        ]

        # Surface connectivity: 1 is positive & -1 is negative orientation, "1_5" is line label
        self.solid_surface_connectivity : list  = [
            [[1, "1_5"], [1, "5_6"], [-1, "2_6"], [-1, "1_2"]],
            [[1, "2_6"], [1, "6_7"], [-1, "3_7"], [-1, "2_3"]],
            [[1, "3_7"], [1, "7_8"], [-1, "4_8"], [-1, "3_4"]],
            [[1, "6_10"], [1, "10_11"], [-1, "7_11"], [-1, "6_7"]],
        ]
        self.facesheet_surfaces_connectivity : list = [
            [[1, "5_9"], [1, "9_10"], [-1, "6_10"], [-1, "5_6"]],
            [[1, "7_11"], [1, "11_12"], [-1, "8_12"], [-1, "7_8"]],
        ]
        self.core_end_surface_box : list = [[[4, 4], [5, 5]], [[7, 7], [10, 10]]]
    
    def __repr__(self):
        return Utils.format_repr(self.__dict__)

class MeshParams:
    """
    Represents mesh parameters for a finite element model.
    Stores element type and mesh density, and derives element attributes.
    """

    def __init__(self, element_type: str, mesh_density: float) -> None:
        """
        Initializes the mesh with the specified element type and mesh density.

        Parameters:
            element_type (str): The type of finite element to be used in the mesh.
            mesh_density (float): The density of the mesh, typically defined in meters.
        """
        self.element_type: str = element_type
        self.mesh_density: float = mesh_density

        # derived attributes
        element_attributes: dict[str, dict[str, Any]] = {
            # solid elements
            "C3D20": {
                "ELEMENT_SHAPE": "BLOCK",
                "ELEMENT_ORDER": 2,
                "REDUCED_INTEGRATION": False,
            },
            "C3D20R": {
                "ELEMENT_SHAPE": "BLOCK",
                "ELEMENT_ORDER": 2,
                "REDUCED_INTEGRATION": True,
            },
            "C3D8": {
                "ELEMENT_SHAPE": "BLOCK",
                "ELEMENT_ORDER": 1,
                "REDUCED_INTEGRATION": False,
            },
            "C3D8R": {
                "ELEMENT_SHAPE": "BLOCK",
                "ELEMENT_ORDER": 1,
                "REDUCED_INTEGRATION": True,
            },
            # shell elements
            "S4": {
                "ELEMENT_SHAPE": "QUAD",
                "ELEMENT_ORDER": 1,
                "ELEMENT_ORDER_INCOMPLETE": False,
            },
            "S4R": {
                "ELEMENT_SHAPE": "QUAD",
                "ELEMENT_ORDER": 1,
                "ELEMENT_ORDER_INCOMPLETE": False,
            },
        }
        self.element_shape = element_attributes[self.element_type]["ELEMENT_SHAPE"]
        self.element_order = element_attributes[self.element_type]["ELEMENT_ORDER"]
        self.reduced_integration = element_attributes[self.element_type]["REDUCED_INTEGRATION"]

    def __repr__(self) -> str:
        return Utils.format_repr(fields=self.__dict__)


class PanelModel:
    def __init__(
        self,
        variables : PanelVariables,
        meshparams : MeshParams
    ) -> None:
        # Geometry
        self.variables= variables
        self.derivedvariables= DerivedPanelVariables(variables=self.variables)

        # Mesh
        self.meshparams = meshparams
        self.generate_mesh()

    def __repr__(self) -> str:
        return Utils.format_repr(fields=self.__dict__)

    def generate_mesh(self) -> None:
        """
        Generates the geometry and mesh for the panel using GMSH.
        Steps:
            1. Builds panel geometry from variables, derived variables and mesh parameters.
            2. Generates mesh, supporting linear/quadratic block elements.
            4. Applies mirroring and patterning for tessellation.
            5. Defines physical groups for volumes and surfaces.
            6. Removes duplicates, renumbers, and saves mesh as .inp.
        """

        def create_surface(connectivity: list, lines: dict) -> int:
            """
            Creates a surface by connecting curves based on the provided connectivity information.

            Args:
                connectivity (list of tuple): A list where each tuple contains information to identify and orient curves.
                Each tuple is expected to have two elements:
                    - The orientation(int)
                    - The curve identifier (str)

            Returns:
                int: The tag or identifier of the created surface.

            Notes:
                - Applies transfinite meshing to the created surface.
            """
            curve_ids: list[int] = [info[0] * lines[info[1]] for info in connectivity]
            loop: int = CAD.addCurveLoop(curve_ids)
            surf_tag: int = CAD.addPlaneSurface([loop])
            CAD.synchronize()
            MESH.setTransfiniteSurface(surf_tag)
            return surf_tag

        def create_solid(surfaces: list, extrude_height: float, num_layers: int) -> list[int]:
            """
            Builds a solid by extruding given surfaces to a specified height and number of layers.

            Args:
                surfaces (array-like): The surfaces to be extruded.
                extrude_height (float): The height to extrude the surfaces.
                num_layers (int): The number of layers in the extrusion.

            Returns:
                list: The tags of the generated solid surfaces.

            Notes:
                - Sets transfinite meshing on the resulting solid surfaces.
            """
            solid = np.asarray(
                CAD.extrude(
                    surfaces,
                    0.0,
                    0.0,
                    extrude_height,
                    numElements=[num_layers],
                    recombine=True,
                )
            )
            solid_tags: list[int] = solid[solid[:, 0] == 3, 1].tolist()
            CAD.synchronize()
            np.vectorize(MESH.setTransfiniteSurface)(solid_tags)
            return solid_tags

        def create_cad() -> None:

            ###############################################
            # Generate the geometry of the GATOR panel
            ###############################################

            NODES: dict[str, int] = {}
            for label, value in self.derivedvariables.coordinates.items():
                NODES[label] = CAD.addPoint(*value)

            LINES: dict[str, int] = {}
            for line_set in self.derivedvariables.line_connectivity:
                line_lengths: list = [
                    np.linalg.norm(self.derivedvariables.coordinates[str(a)] - self.derivedvariables.coordinates[str(b)])
                    for a, b in line_set
                ]
                num_nodes: int = (
                    int(np.ceil(max(line_lengths) / self.meshparams.mesh_density)) + 1
                )
                for a, b in line_set:
                    line_name = f"{a}_{b}"
                    LINES[line_name] = CAD.addLine(NODES[str(a)], NODES[str(b)])
                    CAD.synchronize()
                    MESH.setTransfiniteCurve(LINES[line_name], num_nodes)

            CORE_SURFACES: list[int] = [
                create_surface(surface, lines=LINES) for surface in self.derivedvariables.solid_surface_connectivity
            ]

            FACESHEET_SURFACES: list[int] = []
            if self.variables.facesheet_thickness != 0.0:
                FACESHEET_SURFACES = [
                    create_surface(surface, LINES)
                    for surface in self.derivedvariables.facesheet_surfaces_connectivity
                ]
                num_layers: int = int(
                    np.ceil(
                        self.variables.facesheet_thickness
                        / self.meshparams.mesh_density
                    )
                )
                FACESHEET_SOLIDS: list[int] = create_solid(
                    list(
                        (2, surface) for surface in FACESHEET_SURFACES + CORE_SURFACES
                    ),
                    self.variables.facesheet_thickness,
                    num_layers,
                )
            else:
                CAD.translate(
                    [(2, s) for s in CORE_SURFACES],
                    dx=0.0,
                    dy=0.0,
                    dz=self.variables.facesheet_thickness,
                )

            CAD.synchronize()

            CORE_EXTRUDED_FACES: list = []
            for nodeLabel in self.derivedvariables.core_end_surface_box:
                min_x: float = self.derivedvariables.coordinates[str(nodeLabel[0][0])][0]
                min_y: float = self.derivedvariables.coordinates[str(nodeLabel[0][1])][1]
                max_x: float = self.derivedvariables.coordinates[str(nodeLabel[1][0])][0]
                max_y: float = self.derivedvariables.coordinates[str(nodeLabel[1][1])][1]
                CORE_EXTRUDED_FACES.extend(
                    MODEL.getEntitiesInBoundingBox(
                        min_x - eps,
                        min_y - eps,
                        -self.variables.core_thickness / 2.0 - eps,
                        max_x + eps,
                        max_y + eps,
                        -self.variables.core_thickness / 2.0 + eps,
                        2,
                    )
                )

            num_layers: int = int(
                np.ceil(
                    (self.variables.core_thickness / 2.0)
                    / self.meshparams.mesh_density
                )
            )
            CORE_SOLIDS: list[int] = create_solid(
                CORE_EXTRUDED_FACES, self.variables.core_thickness / 2.0, num_layers
            )

            CAD.synchronize()
            gmsh.write(f"file.stl")

        def tesselate(mirror_steps: list | None, pattern_steps: list | None) -> None:
            """
            Generates a tessellated mesh by mirroring and patterning a repeating unit of the model.

            Args:
                mirror_steps (list | None): List of mirroring steps, each as [tx, ty, tz], or None to skip mirroring.
                pattern_steps (list | None): List of pattern repetitions along each axis, e.g., [nx, ny], or None to skip patterning.
            """

            def get_entities(dim: int) -> dict:
                """
                Retrieves mesh data for all entities of the MODEL in a given dimension.

                Args:
                    dim (int): The dimension of the entities to retrieve (-1 for all).

                Returns:
                    dict: A dictionary mapping (dim, tag) to a tuple of:
                        - boundary entities,
                        - node data (node tags, coordinates),
                        - element data (element types, element tags, node tags).
                """
                m: dict = {}
                entities = MODEL.getEntities(dim)
                for e in entities:
                    bnd = MODEL.getBoundary([e])
                    nod = MESH.getNodes(e[0], e[1])
                    ele = MESH.getElements(e[0], e[1])
                    m[e] = (bnd, nod, ele)
                return m

            def reorder_nodes(e, m, tx: int, ty: int, tz: int) -> list[np.ndarray]:
                """
                Reorders the nodes of mesh elements according to specified axis transformations to create consistent element orientation when the mesh is mirrored.
                """
                num_elements: int = np.size(m[e][2][1])
                num_nodes: int = np.size(m[e][2][2])
                num_nodes_per_element: int = int(num_nodes / num_elements)

                if num_nodes_per_element == 8:
                    node_index = np.array(
                        [
                            [0, 1, 2, 3],  # -Z corner nodes
                            [4, 5, 6, 7],  # +Z corner nodes
                        ]
                    )
                elif num_nodes_per_element == 20:
                    node_index = np.array(
                        [
                            [0, 1, 2, 3],  # -Z corner nodes
                            [4, 5, 6, 7],  # +Z corner nodes
                            [8, 11, 13, 9],  # -Z edge nodes
                            [16, 18, 19, 17],  # +Z edge nodes
                            [10, 12, 14, 15],  # 0Z edge nodes
                        ]
                    )
                else:
                    raise ValueError(
                        f"Error: Unsupported number of nodes per element: {num_nodes_per_element}"
                    )

                if tx == -1:
                    for count, plane_index in enumerate(node_index):
                        if count in [0, 1, 4]:
                            node_index[count] = np.flip(plane_index)
                        elif count in [2, 3]:
                            node_index[count] = plane_index[[2, 1, 0, 3]]
                        else:
                            raise ValueError(
                                "Error: Unsupported number of node planes per element"
                            )
                if ty == -1:
                    for count, plane_index in enumerate(node_index):
                        if count in [0, 1, 4]:
                            node_index[count] = plane_index[[1, 0, 3, 2]]
                        elif count in [2, 3]:
                            node_index[count] = plane_index[[0, 3, 2, 1]]
                        else:
                            raise ValueError(
                                "Error: Unsupported number of node planes per element"
                            )
                if tz == -1:  #
                    if len(node_index) == 2:
                        node_index = node_index[[1, 0]]
                    elif len(node_index) == 5:
                        node_index = node_index[[1, 0, 3, 2, 4]]
                    else:
                        raise ValueError(
                            "Error: Unsupported number of node planes per element"
                        )

                node_index = np.concatenate(node_index, axis=None)
                if num_nodes_per_element != 8:
                    abaqus_to_gmsh_node_order = [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        11,
                        16,
                        9,
                        17,
                        10,
                        18,
                        19,
                        12,
                        15,
                        13,
                        14,
                    ]
                    node_index = node_index[abaqus_to_gmsh_node_order]

                ordered_nodes: list[np.ndarray] = [
                    m[e][2][2][0][
                        list(
                            element_start_index + ordered_index
                            for ordered_index in node_index
                        )
                    ]
                    for element_start_index in range(
                        0, num_nodes, num_nodes_per_element
                    )
                ]
                ordered_nodes = [np.concatenate(ordered_nodes, axis=None)]

                return ordered_nodes

            def transform(
                m: dict,
                offset_entity: int,
                offset_node: int,
                offset_element: int,
                mirror_step: list | None,
                pattern_step: list | None,
            ) -> None:
                """
                Transforms and adds mesh entities, nodes, and elements to the MODEL and MESH objects with optional mirroring and pattern translation.

                Args:
                    m (dict): Dictionary of mesh entities and their data.
                    offset_entity (int): Offset to apply to entity tags.
                    offset_node (int): Offset to apply to node tags.
                    offset_element (int): Offset to apply to element tags.
                    mirror_step (list | None): List of [tx, ty, tz] for mirroring along axes, or None.
                    pattern_step (list | None): List of [ox, oy] for translation along axes, or None.
                """
                if mirror_step is not None:
                    tx, ty, tz = mirror_step[0], mirror_step[1], mirror_step[2]
                else:
                    tx, ty, tz = 1, 1, 1
                if pattern_step is not None:
                    ox, oy = pattern_step[0], pattern_step[1]
                else:
                    ox, oy = 0, 0

                for e in sorted(m):
                    MODEL.addDiscreteEntity(
                        e[0],
                        e[1] + offset_entity,
                        [
                            (abs(b[1]) + offset_entity) * np.copysign(1, b[1])
                            for b in m[e][0]
                        ],
                    )
                    coords = m[e][1][1]
                    transformed_coords = coords.copy()
                    transformed_coords[0::3] = transformed_coords[0::3] * tx + ox
                    transformed_coords[1::3] = transformed_coords[1::3] * ty + oy
                    transformed_coords[2::3] = transformed_coords[2::3] * tz

                    MESH.addNodes(
                        e[0],
                        e[1] + offset_entity,
                        m[e][1][0] + offset_node,
                        transformed_coords,
                    )

                    nodes = (
                        reorder_nodes(e, m, tx, ty, tz)
                        if all([mirror_step != [0, 0, 0], e[0] == 3])
                        else m[e][2][2]
                    )

                    MESH.addElements(
                        e[0],
                        e[1] + offset_entity,
                        m[e][2][0],
                        [t + offset_element for t in m[e][2][1]],
                        [n + offset_node for n in nodes],
                    )

                MESH.removeDuplicateNodes()

            def mirror(mirror_steps: list) -> None:
                """
                Mirrors mesh entities according to a sequence of mirror steps.

                Args:
                    mirror_steps (list): A list of lists, each specifying the mirror transformation for [x, y, z] axes.
                """
                m = get_entities(-1)
                for count, mirror_step in enumerate(mirror_steps, start=1):
                    transform(
                        m,
                        int(count * 10**3),
                        int(count * 10**6),
                        int(count * 10**5),
                        mirror_step,
                        None,
                    )
                MESH.renumberNodes()
                MESH.renumberElements()

            def pattern(pattern_steps: list) -> None:
                """
                Applies a pattern to unit cell model along specified axes to create a panel.
                Args:
                    pattern_steps (list): A list of integers specifying the number of pattern repetitions along each axis (e.g., [nx, ny]).
                """
                xmin, ymin, _, xmax, ymax, _ = MODEL.getBoundingBox(-1, -1)
                axis_lengths: list[float] = [xmax - xmin, ymax - ymin]

                for axis_index, axis_units in enumerate(pattern_steps):
                    axis_length = axis_lengths[axis_index]
                    m = get_entities(-1)

                    for count in range(1, axis_units):
                        quo, rem = divmod(count, 2)
                        offset_length = (
                            (quo + rem) * axis_length * (-1) ** (count - 1)
                        )
                        transform(
                            m,
                            int(count * 10 ** (4 + axis_index)),
                            int(count * 10 ** (7 + axis_index)),
                            int(count * 10 ** (6 + axis_index)),
                            None,
                            [
                                (
                                    offset_length if axis_index == 0 else 0
                                ),
                                offset_length if axis_index == 1 else 0,
                            ],
                        )

                    MESH.renumberNodes()
                    MESH.renumberElements()

            if mirror_steps is not None:
                mirror(mirror_steps)

            if pattern_steps is not None:
                pattern(pattern_steps)

        def create_physical_groups():
            """
            Creates named physical groups for further processing (e.g., boundary conditions).
            Physical groups are created for:
                - FACESHEET_TOP and FACESHEET_BOTTOM (volume, named 'MAT-2')
                - CORE (volume, named 'MAT-1')
                - BOTTOM_FACE (surface, named 'ZNEG')
                - TOP_FACE (surface, named 'ZPOS')
                - RIGHT_FACE (surface, named 'XPOS')
                - REAR_FACE (surface, named 'YPOS')
                - LEFT_FACE (surface, named 'XNEG')
                - FRONT_FACE (surface, named 'YNEG')
            """
            
            # creating required phycal groups
            xmin, ymin, zmin, xmax, ymax, zmax = MODEL.getBoundingBox(-1,-1)
            FACESHEET_TOP=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), ((zmax-self.variables.facesheet_thickness)-eps),
                (xmax+eps), (ymax+eps), (zmax+eps),
                3 
            )
            FACESHEET_BOTTOM=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (zmin-eps),
                (xmax+eps), (ymax+eps), ((zmin+self.variables.facesheet_thickness)+eps),
                3 
            )
            CORE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (-self.variables.core_thickness/2.0-eps),
                (xmax+eps), (ymax+eps), (self.variables.core_thickness/2.0+eps),
                3
            )
            TOP_FACE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (zmax-eps),
                (xmax+eps), (ymax+eps), (zmax+eps),
                2 
            )
            BOTTOM_FACE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (zmin-eps),
                (xmax+eps), (ymax+eps), (zmin+eps),
                2 
            )
            LEFT_FACE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (zmin-eps),
                (xmin+eps), (ymax+eps), (zmax+eps),
                2 
            )
            RIGHT_FACE=MODEL.getEntitiesInBoundingBox(
                (xmax-eps), (ymin-eps), (zmin-eps),
                (xmax+eps), (ymax+eps), (zmax+eps),
                2 
            )
            FRONT_FACE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymin-eps), (zmin-eps),
                (xmax+eps), (ymin+eps), (zmax+eps),
                2 
            )
            REAR_FACE=MODEL.getEntitiesInBoundingBox(
                (xmin-eps), (ymax-eps), (zmin-eps),
                (xmax+eps), (ymax+eps), (zmax+eps),
                2 
            )

            MODEL.addPhysicalGroup(3, list(item[1] for item in FACESHEET_TOP+FACESHEET_BOTTOM), name='MAT-2') # FACESHEET: MAT-2
            MODEL.addPhysicalGroup(3, list(item[1] for item in CORE), name='MAT-1') # CORE: MAT-1
            MODEL.addPhysicalGroup(2, list(item[1] for item in BOTTOM_FACE), name='ZNEG') # FACE_1 # numbering based on element node ordering
            MODEL.addPhysicalGroup(2, list(item[1] for item in TOP_FACE), name='ZPOS') # FACE_2
            MODEL.addPhysicalGroup(2, list(item[1] for item in RIGHT_FACE), name='XPOS') # FACE_3
            MODEL.addPhysicalGroup(2, list(item[1] for item in REAR_FACE), name='YPOS') # FACE_4
            MODEL.addPhysicalGroup(2, list(item[1] for item in LEFT_FACE), name='XNEG') # FACE_5
            MODEL.addPhysicalGroup(2, list(item[1] for item in FRONT_FACE), name='YNEG') # FACE_6
            CAD.synchronize()

        def create_mesh() -> None:
            """
            Generate the mesh along with the mirror and pattern steps required for the panel model, and saves it in ABAQUS format.
            """

            # Mesh generation
            if self.meshparams.element_shape == "BLOCK":
                gmsh.option.setNumber("Mesh.RecombineAll", 1)
                if self.meshparams.element_order == 2:
                    gmsh.option.setNumber("Mesh.ElementOrder", 2)
                    gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
                MESH.generate(3)

            # Mirroring repeating unit to create unit cell
            mirror_steps: list[list[int]] = [
                [1, 1, -1],  # Z
                [-1, 1, 1],  # X
                [-1, 1, -1],  # X and Z
                [1, -1, 1],  # Y
                [1, -1, -1],  # Y and Z
                [-1, -1, 1],  # X and Y
                [-1, -1, -1],  # X, Y and Z
            ]
            tesselate(mirror_steps, None)
            CAD.synchronize()

            # Tesselating unit cell to create panel
            nl = getattr(self.variables, "nl", [1,1])
            if nl[0] > 1 or nl[1] > 1:
                tesselate(None, [nl[0], nl[1]])
            CAD.synchronize()

            # Define physical groups
            create_physical_groups()
            
            # Printing options
            gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -1000)
            gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -100)

            # Saving geometry mesh file
            MESH.removeDuplicateNodes()
            MESH.renumberNodes()
            MESH.renumberElements()
            CAD.synchronize()
            gmsh.write("file.inp")

        # NOTE: Currently not used
        def save_cad(CORE_SOLIDS: list[int], FACESHEET_SOLIDS: list[int]) -> None:
            """
            Exports the combined CAD geometry of core and facesheet solids to an STL file. This function fuses the list of core and facesheet solids into single solids, synchronizes the CAD model, and writes the resulting geometry to a file .
            """
            CORE_SOLIDS = CAD.fuse(
                [(3, CORE_SOLIDS[0])], [(3, tag) for tag in CORE_SOLIDS[1:]]
            )[0][0][1]
            FACESHEET_SOLIDS = CAD.fuse(
                [(3, FACESHEET_SOLIDS[0])],
                [(3, tag) for tag in FACESHEET_SOLIDS[1:]],
            )[0][0][1]
            CAD.synchronize()
            gmsh.write(f"file.stl")
           
        ##########################################
        # Run GMSH to great panel geometry
        #########################################

        gmsh.initialize()

        eps = float(np.finfo(np.float32).eps.item())  # machine epsilon
        gmsh.option.setNumber(
            "General.Verbosity", 2
        )  # Reduce verbosity, print error and warnings

        MODEL = gmsh.model
        CAD = MODEL.occ
        MESH = MODEL.mesh

        # GMSH processes
        create_cad()
        create_mesh()

        # # Show the model
        # gmsh.fltk.run()

        gmsh.finalize()





if __name__ == "__main__":

    panel = PanelVariables(
        core_type="ZPR",
        mql = [True, True, True], 
        nl=[2,1],
        theta=np.deg2rad(60.0),
        chevron_wall_length=Units.mm2m(12.5),
        chevron_thickness=Units.mm2m(1.0),
        chevron_separation=Units.mm2m(6.0),
        rib_thickness=Units.mm2m(1.0),
        core_thickness=Units.mm2m(11.0),
        facesheet_thickness=Units.mm2m(0.8),
    )

    mesh_params = MeshParams(element_type="C3D8R", mesh_density=Units.mm2m(1.0))
    print(PanelModel(variables=panel, meshparams=mesh_params))

    # design = GATOR(
    #     Variables(
    #         core_type="ZPR",
    #         mql = [2,2,2], 
    #         nl=[1,1],
    #         theta=np.deg2rad(60.0),
    #         chevron_wall_length=Units.mm2m(12.5),
    #         chevron_thickness=Units.mm2m(1.0),
    #         chevron_separation=Units.mm2m(6.0),
    #         rib_thickness=Units.mm2m(1.0),
    #         core_thickness=Units.mm2m(11.0),
    #         facesheet_thickness=Units.mm2m(0.8),
    #     ),
    #     MeshParams(element_type="C3D8R", mesh_density=Units.mm2m(1.0)),
    # )

    # print(design.variables)
    # print(design.derivedvariables)
    # print(design.meshparams)
