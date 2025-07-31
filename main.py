"""
Author: Nuhaadh Mahid (nuhaadh.mahid@bristol.ac.uk)
This code is part of the work described in:
N. M. Mahid, M. Schenk, B. Titurus, and B. K. S. Woods, “Parametric design studies of GATOR morphing fairings for folding wingtip joints,” Smart Materials and Structures, vol. 34, no. 2, p. 25049, Jan. 2025, doi: 10.1088/1361-665x/adad21.
"""

import os
import pickle
import json
import time
import traceback
import threading
from typing import Any, Callable
import numpy as np
from itertools import product
import functools
import gmsh
import subprocess
import shutil

class Utils:
    """
    A utility class providing methods used for miscellaneous tasks.
    """

    @staticmethod
    def logger(func: Callable, log_file: str = "logger.txt") -> Callable:
        """
        Decorator to log function execution time and arguments.
        Shows progress if 'debugging' is True.
        """

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            with open(log_file, "a", encoding="utf-8") as log:
                log.write(f"{time.strftime('%d/%m/%Y %H:%M')} :: Running {func.__name__} args: {args}, {kwargs}\n")
            stop_event = threading.Event() if globals().get("debugging", False) else None
            thread = None

            def progress():
                while stop_event is not None and not stop_event.is_set():
                    elapsed = time.time() - start
                    print(f"\rRunning {func.__name__}... {time.strftime('%H:%M:%S', time.gmtime(elapsed))}", end="", flush=True)
                    stop_event.wait(1)

            if stop_event:
                thread = threading.Thread(target=progress)
                thread.start()

            status = "Unknown"
            try:
                result = func(*args, **kwargs)
                status = "Completed"
            except Exception as e:
                result = None
                status = f"Failed: {e}"
                if globals().get("debugging", False):
                    print(traceback.format_exc())
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(traceback.format_exc())
            finally:
                if stop_event and thread:
                    stop_event.set()
                    thread.join()
                duration = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))
                if globals().get("debugging", False):
                    print(f"\r{status} {func.__name__} in {duration}           ")
                with open(log_file, "a", encoding="utf-8") as log:
                    log.write(f"{time.strftime('%d/%m/%Y %H:%M')} :: {status} {func.__name__} in {duration}\n")
            return result

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

    class Directory:
        """
        A class representing a directory structure for an analysis case.
        Contains paths for code, case, and run directories.
        """

        def __init__(self, case_name: str):
            """
            Initializes a Directory instance.

            Args:
                case_name (str): The name of the analysis case.

            Returns:
                Directory: An instance of the Directory class.
            """
            self.run_folder = os.path.abspath(os.getcwd())
            self.case_folder = os.path.join(self.run_folder, case_name)
            self.abaqus_folder = os.path.join(self.run_folder, "abaqus_temp")
            self._create_directories()

        def __repr__(self):
            return Utils.format_repr(fields=self.__dict__)

        def _create_directories(self):
            """Creates the necessary directory structure for the analysis case."""
            for folder in [self.case_folder, self.abaqus_folder]:
                os.makedirs(folder, exist_ok=True)

            subfolders = ["mesh", "input", "inp", "msg", "dat", "odb", "report", "data", "fig", "log", "trace", "sta", "temp"]
            for sub in subfolders:
                os.makedirs(os.path.join(self.case_folder, sub), exist_ok=True)

            for sub in ["inp", "data"]:
                os.makedirs(os.path.join(self.case_folder, sub, "serialised"), exist_ok=True)

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

class PanelDefaults:
    """Centralized default values for panel variables and mesh parameters."""
    
    # IndependentPanelVariables defaults
    core_type = "ZPR" # zero-poisson ratio core
    mql = [True, True] # cell symmetry across x and y directions
    nl = [1, 1] # number of cells in x and y directions
    theta = np.deg2rad(60.0) # chevron angle in radians
    chevron_wall_length = Utils.Units.mm2m(12.5) # length of chevron wall
    chevron_thickness = Utils.Units.mm2m(1.0) # thickness of chevron wall
    chevron_separation = Utils.Units.mm2m(6.0) # separation between chevrons
    rib_thickness = Utils.Units.mm2m(1.0) # thickness of ribs
    core_thickness = Utils.Units.mm2m(11.0) # thickness of core
    facesheet_thickness = Utils.Units.mm2m(0.8) # thickness of facesheet
    core_material = (396.0 * (10 ** 6.0), 0.48) # core material properties (E, nu)
    facesheet_material = (22.9 * (10 ** 6.0), 0.48) # facesheet material properties (E, nu)

    # PanelMeshParams defaults
    element_type = "C3D8R"
    mesh_density = Utils.Units.mm2m(1.0)

class IndependentPanelVariables:
    """
    A class defining GATOR panel geometry variables.
    """
    def __init__(
        self,
        core_type: str | None = None,
        mql: list | None = None,
        nl: list | None = None,
        theta: float | None = None,
        chevron_wall_length: float | None = None,
        chevron_thickness: float | None = None,
        chevron_separation: float | None = None,
        rib_thickness: float | None = None,
        core_thickness: float | None = None,
        facesheet_thickness: float | None = None,
        core_material: tuple | None = None,
        facesheet_material: tuple | None = None
    ) -> None:
        for attr in [
            "core_type", "mql", "nl", "theta", "chevron_wall_length", "chevron_thickness",
            "chevron_separation", "rib_thickness", "core_thickness", "facesheet_thickness",
            "core_material", "facesheet_material"
        ]:
            setattr(
                self,
                attr,
                (
                    locals()[attr] if locals()[attr] is not None
                    else getattr(PanelDefaults, attr)
                ),
            )

    @property
    def core_type(self) -> str:
        return self._core_type

    @core_type.setter
    def core_type(self, value):
        if not isinstance(value, str):
            raise TypeError("core_type must be a string")
        self._core_type = value

    @property
    def mql(self) -> list:
        return self._mql

    @mql.setter
    def mql(self, value):
        if not (isinstance(value, list) and all(isinstance(v, bool) for v in value) and len(value) == 2):
            raise TypeError("mql must be a list of two bools")
        self._mql = value

    @property
    def nl(self) -> list:
        return self._nl

    @nl.setter
    def nl(self, value):
        if not (isinstance(value, list) and all(isinstance(v, int) for v in value) and len(value) == 2):
            raise TypeError("nl must be a list of two ints")
        self._nl = value

    @property
    def theta(self) -> float:
        return self._theta

    @theta.setter
    def theta(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("theta must be a float")
        self._theta = float(value)

    @property
    def chevron_wall_length(self) -> float:
        return self._chevron_wall_length

    @chevron_wall_length.setter
    def chevron_wall_length(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("chevron_wall_length must be a float")
        self._chevron_wall_length = float(value)

    @property
    def chevron_thickness(self) -> float:
        return self._chevron_thickness

    @chevron_thickness.setter
    def chevron_thickness(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("chevron_thickness must be a float")
        self._chevron_thickness = float(value)

    @property
    def chevron_separation(self) -> float:
        return self._chevron_separation

    @chevron_separation.setter
    def chevron_separation(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("chevron_separation must be a float")
        self._chevron_separation = float(value)

    @property
    def rib_thickness(self) -> float:
        return self._rib_thickness

    @rib_thickness.setter
    def rib_thickness(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("rib_thickness must be a float")
        self._rib_thickness = float(value)

    @property
    def core_thickness(self) -> float:
        return self._core_thickness

    @core_thickness.setter
    def core_thickness(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("core_thickness must be a float")
        self._core_thickness = float(value)

    @property
    def facesheet_thickness(self) -> float:
        return self._facesheet_thickness

    @facesheet_thickness.setter
    def facesheet_thickness(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("facesheet_thickness must be a float")
        self._facesheet_thickness = float(value)

    @property
    def core_material(self) -> tuple:
        return self._core_material

    @core_material.setter
    def core_material(self, value: tuple):
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (float, int)) for v in value)):
            raise TypeError("core_material must be a tuple of (E, nu)")
        self._core_material = value

    @property
    def facesheet_material(self) -> tuple:
        return self._facesheet_material

    @facesheet_material.setter
    def facesheet_material(self, value: tuple):
        if not (isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, (float, int)) for v in value)):
            raise TypeError("facesheet_material must be a tuple of (E, nu)")
        self._facesheet_material = value

    def __repr__(self):
        return Utils.format_repr(self.__dict__)

class SolidMeshParams:
    """
    Represents mesh parameters for a finite element model.
    Stores element type and mesh density, and derives element attributes.
    """

    def __init__(
        self,
        element_type: str | None = None,
        mesh_density: float | None = None,
    ) -> None:
        for attr in ["element_type", "mesh_density"]:
            setattr(
                self,
                attr,
                (
                    locals()[attr] if locals()[attr] is not None
                    else getattr(PanelDefaults, attr)
                ),
            )

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

    @property
    def element_type(self) -> str:
        return self._element_type

    @element_type.setter
    def element_type(self, value):
        element_types = ["C3D20", "C3D20R", "C3D8", "C3D8R"]
        if not (isinstance(value, str) and value in element_types):
            raise TypeError("element_type must be a string and one of: " + ", ".join(element_types))
        self._element_type = value

    @property
    def mesh_density(self) -> float:
        return self._mesh_density

    @mesh_density.setter
    def mesh_density(self, value):
        if not isinstance(value, (float, int)):
            raise TypeError("mesh_density must be a float")
        self._mesh_density = float(value)

    def __repr__(self) -> str:
        return Utils.format_repr(fields=self.__dict__)

class DependentPanelVariables:

    def __init__(
        self, 
        variables : IndependentPanelVariables
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
        This method calculates derived variables for the GatorPanel based on the current values in `self.variables`. 
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

class PanelModel:
    def __init__(
        self,
        variables : IndependentPanelVariables = IndependentPanelVariables()
    ) -> None:
        """Initializes the PanelModel with independent variables."""
        # Geometry
        self.variables = variables

    def __repr__(self) -> str:
        return Utils.format_repr(fields=self.__dict__)

    def Analysis(self, 
            directory: Utils.Directory,
            case_number: int = 0,
            meshparams: SolidMeshParams = SolidMeshParams()
        ) -> None:
        """
        Sets up directory structure, initializes mesh parameters, generates the GMSH panel geometry and mesh, and runs homogenisation process.
        """
        # Directory
        self.directory: Utils.Directory = directory
        self.case_number: int = case_number

        # Derived variables
        self.derivedvariables: DependentPanelVariables = DependentPanelVariables(variables=self.variables)

        # Mesh parameters
        self.meshparams: SolidMeshParams = meshparams

        # Combine independent variables, dependent variables, and mesh parameters into a dictionary
        panel_data = {
            **{k.lstrip('_'): v for k, v in self.variables.__dict__.items()},
            **{k.lstrip('_'): v for k, v in self.meshparams.__dict__.items()},
            "qlx": self.derivedvariables.qlx,
            "qly": self.derivedvariables.qly,
            "qlz": self.derivedvariables.qlz,
            "lx": self.derivedvariables.qlx * (int(self.variables.mql[0]) + 1) * int(self.variables.nl[0]),
            "ly": self.derivedvariables.qly * (int(self.variables.mql[1]) + 1) * int(self.variables.nl[1]),
            "area_ratio": self.derivedvariables.core_area_ratio
        }
        # Save the dictionary as a pickle file in the input folder
        Utils.save_object(panel_data, os.path.join(self.directory.case_folder, "input", f"{self.case_number}_panel_data"), method="json")

        # Generate the GMSH panel geometry and mesh
        self.PanelGMSH(self)

        # Homogenisation
        self.PanelHomogenisation(self)

    class PanelGMSH:
        """
        A class to manage GMSH operations for generating the geometry and mesh of a panel.
        This class is responsible for creating the CAD model, generating the mesh, and applying tessellation.
        """

        def __init__(self, panel: "PanelModel") -> None:
            self.panel = panel
            self.generate_mesh()

        @Utils.logger
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
                CAD.mesh.setTransfiniteSurface(surf_tag)
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
                np.vectorize(CAD.mesh.setTransfiniteSurface)(solid_tags)
                return solid_tags

            def create_cad() -> None:

                ###############################################
                # Generate the geometry of the GATOR panel
                ###############################################

                NODES: dict[str, int] = {}
                for label, value in self.panel.derivedvariables.coordinates.items():
                    NODES[label] = CAD.addPoint(*value)

                LINES: dict[str, int] = {}
                for line_set in self.panel.derivedvariables.line_connectivity:
                    line_lengths: list = [
                        np.linalg.norm(self.panel.derivedvariables.coordinates[str(a)] - self.panel.derivedvariables.coordinates[str(b)])
                        for a, b in line_set
                    ]
                    num_nodes: int = (
                        int(np.ceil(max(line_lengths) / self.panel.meshparams.mesh_density)) + 1
                    )
                    for a, b in line_set:
                        line_name = f"{a}_{b}"
                        LINES[line_name] = CAD.addLine(NODES[str(a)], NODES[str(b)])
                        CAD.synchronize()
                        CAD.mesh.setTransfiniteCurve(LINES[line_name], num_nodes)

                CORE_SURFACES: list[int] = [
                    create_surface(surface, lines=LINES) for surface in self.panel.derivedvariables.solid_surface_connectivity
                ]

                FACESHEET_SURFACES: list[int] = []
                if self.panel.variables.facesheet_thickness != 0.0:
                    FACESHEET_SURFACES = [
                        create_surface(surface, LINES)
                        for surface in self.panel.derivedvariables.facesheet_surfaces_connectivity
                    ]
                    num_layers: int = int(
                        np.ceil(
                            self.panel.variables.facesheet_thickness
                            / self.panel.meshparams.mesh_density
                        )
                    )
                    FACESHEET_SOLIDS: list[int] = create_solid(
                        list(
                            (2, surface) for surface in FACESHEET_SURFACES + CORE_SURFACES
                        ),
                        self.panel.variables.facesheet_thickness,
                        num_layers,
                    )
                else:
                    CAD.translate(
                        [(2, s) for s in CORE_SURFACES],
                        dx=0.0,
                        dy=0.0,
                        dz=self.panel.variables.facesheet_thickness,
                    )

                CAD.synchronize()

                CORE_EXTRUDED_FACES: list = []
                for nodeLabel in self.panel.derivedvariables.core_end_surface_box:
                    min_x: float = self.panel.derivedvariables.coordinates[str(nodeLabel[0][0])][0]
                    min_y: float = self.panel.derivedvariables.coordinates[str(nodeLabel[0][1])][1]
                    max_x: float = self.panel.derivedvariables.coordinates[str(nodeLabel[1][0])][0]
                    max_y: float = self.panel.derivedvariables.coordinates[str(nodeLabel[1][1])][1]
                    CORE_EXTRUDED_FACES.extend(
                        MODEL.getEntitiesInBoundingBox(
                            min_x - eps,
                            min_y - eps,
                            -self.panel.variables.core_thickness / 2.0 - eps,
                            max_x + eps,
                            max_y + eps,
                            -self.panel.variables.core_thickness / 2.0 + eps,
                            2,
                        )
                    )

                num_layers: int = int(
                    np.ceil(
                        (self.panel.variables.core_thickness / 2.0)
                        / self.panel.meshparams.mesh_density
                    )
                )
                CORE_SOLIDS: list[int] = create_solid(
                    CORE_EXTRUDED_FACES, self.panel.variables.core_thickness / 2.0, num_layers
                )

                CAD.synchronize()

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
                            int(count * 10**4),
                            int(count * 10**7),
                            int(count * 10**6),
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
                                int(count * 10 ** (5 + axis_index)),
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
                    (xmin-eps), (ymin-eps), (zmax-self.panel.variables.facesheet_thickness-eps),
                    (xmax+eps), (ymax+eps), (zmax+eps),
                    3 
                )
                FACESHEET_BOTTOM=MODEL.getEntitiesInBoundingBox(
                    (xmin-eps), (ymin-eps), (zmin-eps),
                    (xmax+eps), (ymax+eps), (zmin+self.panel.variables.facesheet_thickness+eps),
                    3 
                )
                CORE=MODEL.getEntitiesInBoundingBox(
                    (xmin-eps), (ymin-eps), (-self.panel.variables.core_thickness/2.0-eps),
                    (xmax+eps), (ymax+eps), (self.panel.variables.core_thickness/2.0+eps),
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

                MODEL.addPhysicalGroup(3, list(item[1] for item in FACESHEET_TOP+FACESHEET_BOTTOM), name='FACESHEET') # FACESHEET: MAT-2
                MODEL.addPhysicalGroup(3, list(item[1] for item in CORE), name='CORE') # CORE: MAT-1
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
                if self.panel.meshparams.element_shape == "BLOCK":
                    gmsh.option.setNumber("Mesh.RecombineAll", 1)
                    if self.panel.meshparams.element_order == 2:
                        gmsh.option.setNumber("Mesh.ElementOrder", 2)
                        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
                    MESH.generate(3)

                # Mirroring repeating unit to create unit cell
                mirror_steps: list[list[int]] = []
                # Generate all mirror combinations for enabled axes, except the identity
                axes = self.panel.variables.mql + [True] # Z axis always mirrored
                # For each axis, use [-1, 1] if enabled, else [1]
                options = [[-1, 1] if en else [1] for en in axes]
                for mirror in product(*options):
                    if mirror != (1, 1, 1):
                        mirror_steps.append(list(mirror))
                # Create mirroed mesh
                tesselate(mirror_steps, None)
                CAD.synchronize()

                # Tesselating unit cell to create panel
                nl = getattr(self.panel.variables, "nl", [1,1])
                if nl[0] > 1 or nl[1] > 1:
                    tesselate(None, [nl[0], nl[1]])
                CAD.synchronize()

                # Define physical groups
                create_physical_groups()

                # Printing options
                gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -1000)
                gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -100)

                # Saving geometry mesh file
                gmsh.write(os.path.join(self.panel.directory.case_folder, "mesh", f"{self.panel.case_number}.inp"))

            ##########################################
            # Run GMSH to great panel geometry
            #########################################

            gmsh.initialize()

            eps = np.finfo(np.float32).eps # machine epsilon
            gmsh.option.setNumber(
                "General.Verbosity", 2
            )  # Reduce verbosity, print error and warnings

            MODEL = gmsh.model
            CAD = MODEL.geo
            MESH = MODEL.mesh

            # GMSH processes
            create_cad()
            create_mesh()

            # Show the model
            if gmsh_popup: 
                gmsh.fltk.run()

            gmsh.finalize()

    class PanelHomogenisation:
        def __init__(self, panel: "PanelModel"):
            """Main method to generate the Abaqus input file."""
            self.panel = panel
            self.panel_mesh = self.read_mesh_data(
                os.path.join(self.panel.directory.case_folder, "mesh", f"{self.panel.case_number}.inp"),
                panel.meshparams.element_type
            )
            self.center_and_renumber_mesh()
            self.add_pin_and_control_nodes()

            self.consistent_node_pairs = self.find_node_pairs()
            self.equations = self.generate_constraint_equations() 

            self.write_input_file(panel, self.panel_mesh, equations=self.equations) 

            self.run_abaqus()

        def read_mesh_data(self, mesh_file: str, element_type: str) -> dict:
            """Reads node, element, and set data from a Gmsh-generated .inp file."""
            with open(mesh_file, "r") as file:
                lines = file.read().splitlines()

            panel_mesh = {"nodes": [], "elements": [], "element_type": element_type, "elsets": {}, "nsets": {}}

            num_nodes_per_element = int(element_type.strip('C3D').strip('R')) + 1
            temp_list = []
            section = None
            current_set = None
            for line in lines:
                if line.startswith('*NODE'):
                    section = "nodes"
                elif line.startswith('*ELEMENT'):
                    section = "elements"
                elif line.startswith('*ELSET,ELSET='):
                    section = "elsets"
                    current_set = line.strip().split('=')[1]
                    panel_mesh["elsets"][current_set] = []
                elif line.startswith('*NSET,NSET='):
                    section = "nsets"
                    current_set = line.strip().split('=')[1]
                    panel_mesh["nsets"][current_set] = []
                elif line.startswith('*'):
                    section = None
                elif section:
                    items = [item.strip() for item in line.split(',') if item.strip()]
                    if section == "nodes":
                        panel_mesh["nodes"].append(list(map(float, items)))
                    elif section == "elements":
                        temp_list.extend(list(map(int, items))) 
                        if len(temp_list) == num_nodes_per_element: # numter of nodes in each element
                            panel_mesh["elements"].append(temp_list) # element number and node numbers
                            temp_list=[]
                    elif section == "elsets":
                        panel_mesh["elsets"][current_set].extend(map(int, items))
                    elif section == "nsets":
                        panel_mesh["nsets"][current_set].extend(map(int, items))

            # Convert lists to numpy arrays
            panel_mesh["nodes"] = np.array(panel_mesh["nodes"], dtype=object) 
            panel_mesh["nodes"][:, 0] = panel_mesh["nodes"][:, 0].astype(np.int32)
            panel_mesh["nodes"][:, 1:] = panel_mesh["nodes"][:, 1:].astype(np.float64)

            panel_mesh["elements"] = np.array(panel_mesh["elements"], dtype=np.int32)

            for key in panel_mesh["elsets"]:
                panel_mesh["elsets"][key] = np.array(panel_mesh["elsets"][key], dtype=np.int32)
            for key in panel_mesh["nsets"]:
                panel_mesh["nsets"][key] = np.array(panel_mesh["nsets"][key], dtype=np.int32)

            return panel_mesh

        def center_and_renumber_mesh(self):
            """Converts data to numpy arrays, renumbers elements, and centers the model."""

            original_element_number = self.panel_mesh["elements"][:, 0].copy()
            self.panel_mesh["elements"][:, 0] = np.arange(1, self.panel_mesh["elements"].shape[0] + 1, dtype=np.int32)
            for key in self.panel_mesh["elsets"]:
                elset_indices = np.searchsorted(original_element_number, self.panel_mesh["elsets"][key])
                self.panel_mesh["elsets"][key] = elset_indices + 1

            min_coords = np.min(self.panel_mesh["nodes"][:, 1:], axis=0)
            max_coords = np.max(self.panel_mesh["nodes"][:, 1:], axis=0)
            center = min_coords + (max_coords - min_coords) / 2.0
            self.panel_mesh["nodes"][:, 1:] -= center

        def add_pin_and_control_nodes(self):
            """Adds special nodes for boundary conditions and control."""
            node_central_idx = np.argmin(np.sum(np.square(self.panel_mesh["nodes"][:, 1:]), axis=1))
            node_central = self.panel_mesh["nodes"][node_central_idx, 0]

            all_set_nodes = np.unique(np.concatenate(list(self.panel_mesh["nsets"].values())))
            if node_central in all_set_nodes:
                raise Exception("Error: central node for pinning is already in another set.")
            self.panel_mesh["nsets"]["PIN-NODE"] = np.array([node_central], dtype=np.int32)

            node_number = self.panel_mesh["nodes"][-1, 0] + 1
            for nset_name in ["STRAIN", "CURVATURE", "SHEAR"]:
                # Add the new node with correct types: [node_number, x, y, z]
                new_node = np.array([node_number, 0.0, 0.0, 0.0], dtype=object)
                new_node[0] = np.int32(node_number)
                new_node[1:] = np.float64(0.0)
                self.panel_mesh["nodes"] = np.vstack((self.panel_mesh["nodes"], new_node))
                self.panel_mesh["nsets"][nset_name] = np.array([node_number], dtype=np.int32)
                node_number += 1

        @Utils.logger
        def find_node_pairs(self) -> np.ndarray:
            """Finds corresponding node pairs on opposite faces for periodic BCs."""
            def pair_nodes(face1_nodes, face2_nodes, sort_coord1, sort_coord2):
                def sort_by_coords(nodes):
                    return nodes[np.lexsort((nodes[:, sort_coord2], nodes[:, sort_coord1]))]
                sorted1 = sort_by_coords(face1_nodes)
                sorted2 = sort_by_coords(face2_nodes)
                return np.column_stack((sorted1[:, 0], sorted2[:, 0])).astype(np.int32)

            node_map = {node[0]: node for node in self.panel_mesh["nodes"]}

            xpos_nodes = np.array([node_map[n] for n in self.panel_mesh["nsets"]["XPOS"]])
            xneg_nodes = np.array([node_map[n] for n in self.panel_mesh["nsets"]["XNEG"]])
            if len(xpos_nodes) != len(xneg_nodes):
                raise ValueError("X+ and X- faces have different node counts.")
            xnode_pairs = pair_nodes(xpos_nodes, xneg_nodes, 2, 3)

            ypos_nodes = np.array([node_map[n] for n in self.panel_mesh["nsets"]["YPOS"]])
            yneg_nodes = np.array([node_map[n] for n in self.panel_mesh["nsets"]["YNEG"]])
            if len(ypos_nodes) != len(yneg_nodes):
                raise ValueError("Y+ and Y- faces have different node counts.")
            ynode_pairs = pair_nodes(ypos_nodes, yneg_nodes, 1, 3)

            if np.unique(xnode_pairs).size != xnode_pairs.size:
                raise Exception("Duplicate node found in X-face pairs.")
            if np.unique(ynode_pairs).size != ynode_pairs.size:
                raise Exception("Duplicate node found in Y-face pairs.")

            edge_nodes = np.intersect1d(self.panel_mesh["nsets"]["XPOS"], self.panel_mesh["nsets"]["YPOS"])
            ynode_pairs_filtered = ynode_pairs[~np.isin(ynode_pairs[:, 0], edge_nodes)]

            return np.vstack((xnode_pairs, ynode_pairs_filtered))

        def generate_constraint_equations(self):  # TODO add transverse shear loading
            """Generates *EQUATION cards for periodic boundary conditions (vectorized for speed)."""
            eps = np.finfo(np.float32).eps
            node_map = {node[0]: node[1:] for node in self.panel_mesh["nodes"]}

            equations = []
            for n1_id, n2_id in self.consistent_node_pairs:
                p1 = node_map[n1_id]
                p2 = node_map[n2_id]

                # Terms for displacement difference equations
                A, C = p1[0] - p2[0], p1[1] - p2[1]
                B, D = A * p1[2], C * p1[2]
                E = -(p1[0]**2 - p2[0]**2) / 2.0
                F = -(p1[1]**2 - p2[1]**2) / 2.0
                G = -(p1[0] * p1[1] - p2[0] * p2[1])

                # Equation for u_x difference
                eq_x = [f"{n1_id}, 1, -1.0", f"{n2_id}, 1, 1.0"]
                if abs(A) > eps: eq_x.append(f"STRAIN, 1, {A}")
                if abs(B) > eps: eq_x.append(f"CURVATURE, 1, {B}")
                if abs(C) > eps: eq_x.append(f"STRAIN, 3, {0.5 * C}")
                if abs(D) > eps: eq_x.append(f"CURVATURE, 3, {0.5 * D}")
                equations.append(eq_x)

                # Equation for u_y difference
                eq_y = [f"{n1_id}, 2, -1.0", f"{n2_id}, 2, 1.0"]
                if abs(A) > eps: eq_y.append(f"STRAIN, 3, {0.5 * A}")
                if abs(B) > eps: eq_y.append(f"CURVATURE, 3, {0.5 * B}")
                if abs(C) > eps: eq_y.append(f"STRAIN, 2, {C}")
                if abs(D) > eps: eq_y.append(f"CURVATURE, 2, {D}")
                equations.append(eq_y)

                # Equation for u_z difference
                eq_z = [f"{n1_id}, 3, -1.0", f"{n2_id}, 3, 1.0"]
                if abs(E) > eps: eq_z.append(f"CURVATURE, 1, {E}")
                if abs(F) > eps: eq_z.append(f"CURVATURE, 2, {F}")
                if abs(G) > eps: eq_z.append(f"CURVATURE, 3, {0.5 * G}")
                equations.append(eq_z)

            return equations

        def write_input_file(self, panel, panel_mesh: dict, equations=None):
            """Writes the complete Abaqus .inp file."""

            def format_lines(data_list, items_per_line=16):
                lines : list[str] = []
                for i in range(0, len(data_list), items_per_line):
                    chunk = data_list[i:i + items_per_line]
                    lines.append(", ".join(map(str, chunk)) + ("," if i + items_per_line < len(data_list) else ""))
                return lines

            lines = ["*NODE"]
            lines.extend([line for node in panel_mesh["nodes"] for line in format_lines(node)])

            lines.append(f"*ELEMENT, TYPE={panel.meshparams.element_type}")
            lines.extend([line for element in panel_mesh["elements"] for line in format_lines(element)])

            for set_name, set_items in panel_mesh["elsets"].items():
                lines.append(f"*ELSET, ELSET={set_name}")
                lines.extend(format_lines(set_items))
            for set_name, set_items in panel_mesh["nsets"].items():
                lines.append(f"*NSET, NSET={set_name}")
                lines.extend(format_lines(set_items))

            lines.extend(["*ORIENTATION, NAME=GLOBAL, DEFINITION=COORDINATES", "1., 0., 0., 0., 1., 0.", "3, 0."])

            # Add section controls for reduced integration elements
            if panel.meshparams.reduced_integration:
                lines.append("*SECTION CONTROLS, NAME=EC-1, HOURGLASS=ENHANCED")

            # Write solid section definitions for each element set
            for set_name in panel_mesh["elsets"]:
                lines.append(f"*SOLID SECTION, ELSET={set_name}, MATERIAL=MAT-{set_name}, ORIENTATION=GLOBAL" + (", CONTROLS=EC-1" if panel.meshparams.reduced_integration else ""))

            # Define materials
            if "CORE" in panel_mesh["elsets"].keys():
                lines.extend(["*MATERIAL, NAME=MAT-CORE", "*ELASTIC", f"{panel.variables.core_material[0]}, {panel.variables.core_material[1]}"])
            if "FACESHEET" in panel_mesh["elsets"].keys():
                lines.extend(["*MATERIAL, NAME=MAT-FACESHEET", "*ELASTIC", f"{panel.variables.facesheet_material[0]}, {panel.variables.facesheet_material[1]}"])

            # Add equations for periodic boundary conditions
            if equations:
                lines.append("*EQUATION")
                for eq in equations:
                    lines.append(str(len(eq)))
                    lines.extend(format_lines(eq, items_per_line=4))

            # Add boundary conditions for homogenisation
            lines.append("*BOUNDARY")
            lines.append("PIN-NODE, PINNED")  # Pin all DOFs at the central node

            # Define homogenisation load cases and associated nodes/DOFs
            load_cases = [
                ("E11", "STRAIN", 1),
                ("E22", "STRAIN", 2),
                ("E12", "STRAIN", 3),
                ("K11", "CURVATURE", 1),
                ("K22", "CURVATURE", 2),
                ("K12", "CURVATURE", 3),
            ] # TODO add transverse shear loading

            output_requested = False
            # Start perturbation step
            lines.append("********************************************** PERTURBATION STEP : 0 START")
            lines.append("*STEP, NAME=HOMOGENISATION-0, PERTURBATION")
            lines.append("*STATIC")
            if not output_requested:
                lines.extend([
                    "*OUTPUT, FIELD",
                    "NODE OUTPUT",
                    "CF, RF, U",
                    "*ELEMENT OUTPUT, DIRECTIONS=YES",
                    "E, LE, S, TRSHR",
                    "*OUTPUT, FIELD, VARIABLE=PRESELECT",
                    "*OUTPUT, HISTORY, FREQUENCY=0"
                ])
                output_requested = True

            # For each load case, apply unit displacement to the corresponding control node/DOF
            for case_name, nset, dof in load_cases:
                lines.append(f"*LOAD CASE, NAME=STATE-0-{case_name}")
                lines.append("*BOUNDARY, OP=NEW") # FIXME this is not correct, but works for now
                lines.append(f"{nset}, {dof}, {dof}, 1.0")
                # All other control nodes fixed (homogenisation: only one DOF active at a time)
                for other_nset, other_dof in [(n, d) for (_, n, d) in load_cases if (n, d) != (nset, dof)]:
                    lines.append(f"{other_nset}, {other_dof}, {other_dof}")
                # lines.append("PIN-NODE, PINNED")  # Ensure pin node remains fixed
                lines.append("*END LOAD CASE")

            lines.append("*END STEP")
            lines.append("********************************************** PERTURBATION STEP : 0 END")

            with open(os.path.join(self.panel.directory.case_folder, "inp", f"{self.panel.case_number}_RVE.inp"), "w") as f:
                f.write("\n".join(lines))

        @Utils.logger
        def run_abaqus(self):
            """Runs the Abaqus analysis using the subprocess module."""

            def run_subprocess(command: str, run_folder: str, log_file: str):
                """Runs a shell command and captures its output."""
                try:
                    process = subprocess.run(
                        command,
                        shell=True,
                        check=True,
                        cwd=run_folder,
                        input='y',
                        text=True,
                        capture_output=True
                    )
                    with open(log_file, 'a') as log:
                        log.write(process.stdout)
                        log.write(process.stderr)
                except subprocess.CalledProcessError as e:
                    print(f"ERROR: Execution failed for command:\n{command}.")
                    print(f"Return code: {e.returncode}")
                    print(f"--- STDOUT ---\n{e.stdout}")
                    print(f"--- STDERR ---\n{e.stderr}")
                    with open(log_file, 'a') as log:
                        log.write(f"ERROR: Execution failed for command:\n{command}.\n")
                        log.write(f"Return code: {e.returncode}\n")
                        log.write(f"--- STDOUT ---\n{e.stdout}\n")
                        log.write(f"--- STDERR ---\n{e.stderr}\n")

            # Job definition
            case_name = f"{self.panel.case_number}_RVE"
            job_file = os.path.join(self.panel.directory.case_folder, "inp", f"{case_name}.inp")
            run_folder = os.path.join(self.panel.directory.abaqus_folder, f"{self.panel.case_number}")
            os.makedirs(run_folder, exist_ok=True)

            # Ensure the log directory exists
            log_file = os.path.join(self.panel.directory.case_folder, "log", f"{self.panel.case_number}_RVE_Abaqus.log")

            # Run the Abaqus analysis
            abaqus_path = "C:\\SIMULIA\\Commands\\abaqus.bat"
            command = f"{abaqus_path} analysis double=both job={case_name} input={job_file} cpus=1 mp_mode=thread interactive"
            run_subprocess(
                command,
                run_folder,
                log_file
            )

            # Copy result files
            for ext in ["odb", "msg", "dat", "sta"]:
                src = os.path.join(run_folder, f"{case_name}.{ext}")
                dst = os.path.join(self.panel.directory.case_folder, ext, f"{case_name}.{ext}")
                try:
                    if os.path.exists(src):
                        shutil.copy(src, dst)
                except Exception:
                    traceback.print_exc()
                    with open(os.path.join(self.panel.directory.case_folder, 'log', f"{self.panel.case_number}_RVE_Abaqus.log"), 'a') as f:
                        f.write(f"ERROR: Cannot copy result file {ext} for case number {self.panel.case_number}\n")

            # Extract results using a Python script
            script_file = os.path.abspath("abaqus_script_to_extract_ABD.py")
            command = f"{abaqus_path} python {script_file} -- {self.panel.directory.case_folder} {self.panel.case_number}"
            run_subprocess(
                command,
                run_folder,
                log_file
            )

class CasesManager:
    """
    A class to manage analysis cases, including their setup and execution.
    """ 
    def __init__(
        self,
        case_name: str = "default_case",
        cases: list[PanelModel] = [],
    ) -> None:
        """
        Initializes the CaseManager with a specified case name.
        Creates necessary directories for the analysis case.

        Parameters:
            case_name (str): The name of the analysis case. Defaults to "default_case".
        """
        self.case_name = case_name
        self.directory = Utils.Directory(case_name=self.case_name)
        if cases != []:
            self.cases = cases
        else:
            self.cases = [PanelModel()]
            
    def __repr__(self) -> str:
        return Utils.format_repr(fields=self.__dict__)


if __name__ == "__main__":

    global debugging, gmsh_popup, logger_path
    debugging : bool = True
    gmsh_popup : bool = False
    
    # default case
    cm = CasesManager("example")

    # Homogenisation
    cm.cases[0].Analysis(directory=cm.directory)
