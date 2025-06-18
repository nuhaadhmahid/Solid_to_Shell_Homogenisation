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
- Use the `mesh_params` class to define mesh settings.
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

import gmsh
import numpy as np


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


class Utils:
    """
    A utility class providing methods used for miscellaneous tasks.
    """

    @staticmethod
    def logger(function):
        """
        A decorator that logs the start and end time of a function execution, along with its arguments.
        It writes the function name, arguments, and the time taken to execute the function to a log file.
        """

        log_file = "logger.txt"

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
    def save_object(obj, path, method):
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
    def load_object(path, method):
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
    def format_repr(fields):
        """
        Formats a list/array of (name, value, fmt, unit) tuples for __repr__ display.
        """
        max_name_length = 20
        max_value_length = 10
        lines = [f"{'':-^{max_name_length+max_value_length+3}}"]  # title line
        for name, value in fields.items():
            value_type = type(value)
            if value_type in (np.float64, np.float32, float):
                value_str = f"{value:.4f}"
            elif value_type in (np.int64, np.int32, int):
                value_str = f"{value:.0f}"
            elif value_type in (np.bool_, bool):
                value_str = str(value)
            else:
                value_str = str(value)  # convert other types to string
            lines.append(f"{name:<{max_name_length}} = {value_str:>{max_value_length}}")
        lines.append(f"{'':-^{max_name_length+max_value_length+3}}")  # end line
        return "\n".join(lines)

    @staticmethod
    def create_case_dir(case_name):
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


class Variables:
    """
    A general-purpose class for storing and representing any set of named variables.
    """

    def __init__(self, **kwargs):
        """
        Initializes the variables object with arbitrary keyword arguments.

        Parameters:
            **kwargs: Arbitrary keyword arguments representing variable names and their values.
        """
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self):
        return Utils.format_repr(self.__dict__)


class MeshParams:
    """
    A class representing a mesh for a finite element model.
    It contains methods to create and manipulate the mesh.
    """

    def __init__(self, element_type, mesh_density):
        """
        Initializes the mesh with the specified element type and mesh density.

        Parameters:
            element_type (str): The type of finite element to be used in the mesh.
            mesh_density (float): The density of the mesh, typically defined in meters.
        """
        self.element_type = element_type
        self.mesh_density = mesh_density

        # derived attributes
        element_attributes = {
            # soldid elements
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
        self.reduced_integration = element_attributes[self.element_type][
            "REDUCED_INTEGRATION"
        ]

    def __repr__(self):
        return Utils.format_repr(self.__dict__)


class GATOR:
    """
    GATOR panel class for generating and meshing a chevron-based core sandwich panel geometry.
    This class encapsulates the geometry definition, derived variable calculation, and mesh generation
    for a GATOR (Geometric Advanced Topology Optimization for RVE) panel. It supports the creation of
    both CAD geometry and finite element meshes using GMSH, including advanced features such as
    mirroring, patterning, and exporting to various formats.
    Attributes:
        variables: An object containing the geometric and panel parameters.
        mesh_params: An object specifying mesh discretization parameters.
        derived_variables: An object containing computed geometric properties based on input variables.
    Methods:
        __init__(variables, mesh_params):
            Initializes the GATOR panel with specified geometry and mesh parameters.
        generate_derived_variables():
        generate_mesh():
            Generates the geometry and mesh for the GATOR panel using GMSH.
            Handles CAD creation, meshing, mirroring, patterning, and exporting to file formats.
    """

    def __init__(self, variables, mesh_params):
        """
        Initializes the GATOR panel class with the given variables and mesh parameters.
        Args:
            variables: The variables to define GATOR panel geometry.
            mesh_params: The mesh parameters to dicretise the RVE.
        """

        self.variables = variables
        self.mesh_params = mesh_params
        self.derived_variables = self.generate_derived_variables()

        self.generate_mesh()  # run gmsh to generate the geometry

    def generate_derived_variables(self):
        """
        Computes and returns derived geometric parameters based on the current set of variables.
        Returns:
            variables: An instance of the `variables` class containing the following derived attributes:
                - qlx (float): Projected chevron wall length in the X direction.
                - qly (float): Projected chevron wall length in the Y direction.
                - LX (float): Total length in the X direction.
                - LY (float): Total length in the Y direction.
                - LZ (float): Total length in the Z direction (core plus facesheets).
                - core_area_ratio (float): Ratio of the core area to the total rectangular area.
        """
        v = self.variables

        si = 0.5 * v.chevron_thickness / np.cos(v.theta) + 0.5 * v.chevron_separation
        so = (
            +0.5 * v.chevron_separation
            + 0.5 * v.chevron_thickness / np.cos(v.theta)
            - 0.5 * v.rib_thickness * np.tan(v.theta)
        )

        qlx = v.chevron_wall_length * np.cos(v.theta)
        qly = si + so + v.chevron_wall_length * np.sin(v.theta)
        qlz = v.core_thickness / 2.0 + v.facesheet_thickness

        core_area_ratio = (
            +qly * v.rib_thickness / 2.0  # rib area
            + (v.chevron_wall_length - (v.rib_thickness / 2.0) / np.cos(v.theta))
            * v.chevron_thickness  # chevron area
        ) / (
            qlx * qly  # total rectangular area
        )

        return Variables(
            si=si,
            so=so,
            qlx=qlx,
            qly=qly,
            qlz=qlz,
            core_area_ratio=core_area_ratio,
        )

    @Utils.logger
    def generate_mesh(self):
        """
        Generates the mesh and geometry for a GATOR panel using GMSH.
        This method orchestrates the creation of a parametric mesh and CAD geometry for a panel structure,
        including core and facesheet solids, using the GMSH Python API. It defines node coordinates,
        line and surface connectivity, builds surfaces and solids, and handles mesh tessellation,
        mirroring, and patterning to create a full panel model. The method supports exporting the geometry
        as STL or the mesh as ABAQUS .inp files.
        The process includes:
            - Defining geometry parameters and mesh connectivity.
            - Building CAD geometry (points, lines, surfaces, solids).
            - Extruding surfaces to create 3D solids for core and facesheet.
            - Applying transfinite meshing for structured mesh generation.
            - Tessellating the mesh by mirroring and patterning to form the complete panel.
            - Exporting the geometry or mesh in the desired format.
        The method relies on instance variables for geometry and mesh parameters, and uses several
        nested helper functions for modularity.
        self : object
            The class instance containing geometry and mesh parameters, as well as mesh control options.
        None
        Raises
        ------
        Exception
            If an unsupported run_type is specified.
        Notes
        -----
        - Requires the GMSH Python API (`gmsh`).
        - The mesh and geometry are generated based on parameters defined in the class instance.
        - The method finalizes the GMSH session at the end.
        """

        # Define variables to be used as nonlocal in nested function
        COORDINATES = None
        LINE_CONNECTIVITY = None
        SOLID_SURFACE_CONNECTIVITY = None
        FACESHEET_SURFACES_CONNECTIVITY = None
        CORE_END_SURFACE_BOX = None

        def generate_geometry_params():
            v = self.variables
            dv = self.derived_variables

            # variables accessible in parent function
            nonlocal COORDINATES, LINE_CONNECTIVITY, SOLID_SURFACE_CONNECTIVITY, FACESHEET_SURFACES_CONNECTIVITY, CORE_END_SURFACE_BOX

            # Define node coordinates
            COORDINATES = {
                "1": [-dv.qlx, 0.0, -dv.qlz],
                "2": [
                    -dv.qlx,
                    -dv.qly
                    + dv.so
                    + v.rib_thickness / 2 * np.tan(v.theta)
                    + v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "3": [
                    -dv.qlx,
                    -dv.qly
                    + dv.so
                    + v.rib_thickness / 2 * np.tan(v.theta)
                    - v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "4": [-dv.qlx, -dv.qly, -dv.qlz],
                "5": [-dv.qlx + v.rib_thickness / 2, 0.0, -dv.qlz],
                "6": [
                    -dv.qlx + v.rib_thickness / 2,
                    -dv.qly
                    + dv.so
                    + v.rib_thickness / 2 * np.tan(v.theta)
                    + v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "7": [
                    -dv.qlx + v.rib_thickness / 2,
                    -dv.qly
                    + dv.so
                    + v.rib_thickness / 2 * np.tan(v.theta)
                    - v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "8": [-dv.qlx + v.rib_thickness / 2, -dv.qly, -dv.qlz],
                "9": [0.0, 0.0, -dv.qlz],
                "10": [
                    0.0,
                    -dv.si + v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "11": [
                    0.0,
                    -dv.si - v.chevron_thickness / 2 / np.cos(v.theta),
                    -dv.qlz,
                ],
                "12": [0.0, -dv.qly, -dv.qlz],
            }
            for key in COORDINATES:
                COORDINATES[key] = np.asarray(COORDINATES[key], dtype=float)

            # Define line connectivity
            LINE_CONNECTIVITY = [
                [[1, 5], [2, 6], [3, 7], [4, 8]],  # horizontal, left to right, column 1
                [
                    [5, 9],
                    [6, 10],
                    [7, 11],
                    [8, 12],
                ],  # horizontal, left to right, column 2
                [[1, 2], [5, 6], [9, 10]],  # veritcal, top to bottom, row 1
                [[2, 3], [6, 7], [10, 11]],  # veritcal, top to bottom, row 2
                [[3, 4], [7, 8], [11, 12]],  # veritcal, top to bottom, row 3
            ]

            # Surface connectivity: 1 is positive & -1 is negative orintation, "1_5" is line label
            SOLID_SURFACE_CONNECTIVITY = [
                [[1, "1_5"], [1, "5_6"], [-1, "2_6"], [-1, "1_2"]],
                [[1, "2_6"], [1, "6_7"], [-1, "3_7"], [-1, "2_3"]],
                [[1, "3_7"], [1, "7_8"], [-1, "4_8"], [-1, "3_4"]],
                [[1, "6_10"], [1, "10_11"], [-1, "7_11"], [-1, "6_7"]],
            ]
            FACESHEET_SURFACES_CONNECTIVITY = [
                [[1, "5_9"], [1, "9_10"], [-1, "6_10"], [-1, "5_6"]],
                [[1, "7_11"], [1, "11_12"], [-1, "8_12"], [-1, "7_8"]],
            ]
            CORE_END_SURFACE_BOX = [[[4, 4], [5, 5]], [[7, 7], [10, 10]]]

        def generate_cad(run_type="inp"):

            # Helper to build surface from connectivity
            def build_surface(connectivity):
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
                curve_ids = [info[0] * LINES[info[1]] for info in connectivity]
                loop = CAD.addCurveLoop(curve_ids)
                surf_tag = CAD.addPlaneSurface([loop])
                CAD.synchronize()
                MESH.setTransfiniteSurface(surf_tag)
                return surf_tag

            # Helper to extrude surfaces
            def build_solid(surfaces, extrude_height, num_layers):
                """
                Builds a solid by extruding given surfaces to a specified height and number of layers.

                Args:
                    surfaces (array-like): The surfaces to be extruded.
                    extrude_height (float): The height to extrude the surfaces.
                    num_layers (int): The number of layers in the extrusion.

                Returns:
                    numpy.ndarray: The tags of the generated solid surfaces.

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
                solid_tags = solid[solid[:, 0] == 3, 1]
                CAD.synchronize()
                np.vectorize(MESH.setTransfiniteSurface)(solid_tags)
                return solid_tags

            # Helper to tessellate the mesh
            def tesselate(mirror_steps, pattern_steps):
                """
                Generates a tessellated mesh by mirroring and patterning a repeating unit of the model.
                This function automates the creation of complex mesh structures by:
                  - Mirroring the unit cell across specified planes to form a full unit cell.
                  - Repeating (patterning) the unit cell along specified axes to create a larger panel.
                The tessellation process involves:
                  - Extracting mesh entities, nodes, and elements.
                  - Optionally mirroring the mesh across user-defined planes.
                  - Optionally repeating the mesh in a pattern along the x and y axes.
                  - Reordering element's nodes for consistent orientation.
                  - Adding new entities, nodes, and elements to the mesh, and removing duplicates.
                    mirror_steps (list of tuple or None):
                        List of mirroring transformations, where each tuple (tx, ty, tz) specifies mirroring along the x, y, and z axes.
                        Use 1 for no mirroring, -1 for mirroring. If None, no mirroring is performed.
                    pattern_steps (list or tuple of int or None):
                        Number of repetitions along each axis, e.g., [nx, ny]. If None, no patterning is performed.
                """

                # Get the mesh data
                def get_entities(dim):
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
                    m = {}
                    entities = MODEL.getEntities(dim)
                    for e in entities:
                        bnd = MODEL.getBoundary([e])
                        nod = MESH.getNodes(e[0], e[1])
                        ele = MESH.getElements(e[0], e[1])
                        m[e] = (bnd, nod, ele)
                    return m

                # Reordering nodes of BLOCK elements for consistent orientation
                def reorder_nodes(e, m, tx, ty, tz):
                    """
                    Reorders the nodes of mesh elements according to specified axis transformations.
                    This function handles the reordering of nodes for BLOCK elements, supporting both 8-node (hexahedral) and 20-node (quadratic hexahedral) elements, and returns the reordered node indices for each element.

                    Parameters
                    ----------
                    e : tuple or list
                        Element descriptor, where e[0] is the element type.
                    m : dict or similar mapping
                        Mesh data structure containing element and node information.
                    tx : int
                        Transformation along the x-axis. Use -1 to flip, 1 for no change.
                    ty : int
                        Transformation along the y-axis. Use -1 to flip, 1 for no change.
                    tz : int
                        Transformation along the z-axis. Use -1 to flip, 1 for no change.
                    Returns
                    -------
                    ordered_nodes : list of numpy.ndarray
                        List containing arrays of reordered node indices for each element.

                    """

                    # numbers of elements and nodes
                    num_elements = np.size(m[e][2][1])
                    num_nodes = np.size(m[e][2][2])
                    num_nodes_per_element = int(num_nodes / num_elements)

                    # node numbering index
                    if num_nodes_per_element == 8:
                        # -Z and +Z corner nodes
                        node_index = np.array(
                            [
                                [0, 1, 2, 3],  # -Z corner nodes
                                [4, 5, 6, 7],  # +Z corner nodes
                            ]
                        )
                    elif num_nodes_per_element == 20:
                        # gmsh node numbering, in abaqus order
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

                    # traformation
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

                    # contracte index to single array and re-ordering array to match gmsh node ordering
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

                    # reordering nodes for each element
                    ordered_nodes = [
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

                # Transform the mesh and create new discrete entities
                def transform(
                    m,
                    offset_entity,
                    offset_node,
                    offset_element,
                    mirror_step,
                    pattern_step,
                ):
                    """
                    Transforms and adds mesh entities, nodes, and elements to the MODEL and MESH objects with optional mirroring and pattern translation.
                    Parameters:
                        m (dict): Dictionary representing the mesh, where keys are entity tuples and values contain entity, node, and element data.
                        offset_entity (int): Offset to apply to entity IDs when adding new entities.
                        offset_node (int): Offset to apply to node IDs when adding new nodes.
                        offset_element (int): Offset to apply to element IDs when adding new elements.
                        mirror_step (tuple or None): Tuple of three values (tx, ty, tz) to mirror coordinates along x, y, z axes. If None, defaults to (1, 1, 1).
                        pattern_step (tuple or None): Tuple of two values (ox, oy) to translate coordinates along x and y axes. If None, defaults to (0, 0).
                    """

                    # Sorting variatble
                    if mirror_step != None:
                        tx, ty, tz = mirror_step[0], mirror_step[1], mirror_step[2]
                    else:
                        tx, ty, tz = 1, 1, 1
                    if pattern_step != None:
                        ox, oy = pattern_step[0], pattern_step[1]
                    else:
                        ox, oy = 0, 0

                    # transforming mesh entities
                    for e in sorted(m):
                        # Add new discrete entity
                        MODEL.addDiscreteEntity(
                            e[0],
                            e[1] + offset_entity,
                            [
                                (abs(b[1]) + offset_entity) * np.copysign(1, b[1])
                                for b in m[e][0]
                            ],
                        )
                        # Transform node coordinates
                        coords = m[e][1][1]
                        transformed_coords = coords.copy()
                        transformed_coords[0::3] = transformed_coords[0::3] * tx + ox
                        transformed_coords[1::3] = transformed_coords[1::3] * ty + oy
                        transformed_coords[2::3] = transformed_coords[2::3] * tz

                        # Add transformed nodes in a single call
                        MESH.addNodes(
                            e[0],
                            e[1] + offset_entity,
                            m[e][1][0] + offset_node,
                            transformed_coords,
                        )

                        # reorder element's node numbering to have a consistent element orientation
                        nodes = (
                            reorder_nodes(e, m, tx, ty, tz)
                            if all([mirror_step != [0, 0, 0], e[0] == 3])
                            else m[e][2][2]
                        )

                        # add elements with transformed nodes
                        MESH.addElements(
                            e[0],
                            e[1] + offset_entity,
                            m[e][2][0],
                            [t + offset_element for t in m[e][2][1]],
                            [n + offset_node for n in nodes],
                        )

                    # Removing duplicate nodes
                    MESH.removeDuplicateNodes()

                # Helper for mirroring the mesh to create a full unit cell
                def mirror(mirror_steps):
                    """
                    Mirrors entities across specified planes and renumbers mesh nodes and elements.
                    Args:
                        mirror_steps (list of tuple): A list where each tuple contains three values (typically representing the normal vector of the mirror plane)
                            to define the mirroring transformation for each step.
                    """

                    # get entities
                    m = get_entities(-1)

                    # mirroring on planes
                    for count, mirror_step in enumerate(mirror_steps, start=1):
                        transform(
                            m,
                            int(count * 10**3),
                            int(count * 10**6),
                            int(count * 10**5),
                            mirror_step,  # mirroring
                            None,  # offseting
                        )

                    # Renumbering
                    MESH.renumberNodes()
                    MESH.renumberElements()

                # Helper for creating a pattern of the unit cell
                def pattern(pattern_steps):
                    """
                    Generates a patterned array of unit cells along specified axes.
                    This function replicates the current unit cell in a pattern defined by `pattern_steps`,
                    which specifies the number of repetitions along each axis. The pattern alternates the
                    direction of offset for each repetition, effectively creating a mirrored and shifted
                    arrangement of the unit cell entities.
                    Args:
                        pattern_steps (list or tuple of int): Number of repetitions along each axis.
                            For example, [3, 2] will repeat the unit cell 3 times along the x-axis and
                            2 times along the y-axis.
                    """

                    # dimensions of the unit cell along each axis
                    xmin, ymin, _, xmax, ymax, _ = MODEL.getBoundingBox(-1, -1)
                    axis_lengths = [xmax - xmin, ymax - ymin]

                    # pattern along each axis
                    for axis_index, axis_units in enumerate(pattern_steps):

                        # axis length for unit cell
                        axis_length = axis_lengths[axis_index]

                        # get unit cell entities
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
                                None,  # mirroring
                                [
                                    (
                                        offset_length if axis_index == 0 else 0
                                    ),  # offseting
                                    offset_length if axis_index == 1 else 0,
                                ],
                            )

                        # Renumbering
                        MESH.renumberNodes()
                        MESH.renumberElements()

                # mirror steps to create the full unit cell
                if mirror_steps is not None:
                    mirror(mirror_steps)

                # pattern the unit cell to create the full panel
                if pattern_steps is not None:
                    pattern(pattern_steps)

            # Helper for saving the geometry to a STEP file
            def export_geometry_unit(CORE_SOLIDS, FACESHEET_SOLIDS, run_type):
                """
                Exports the combined CAD geometry of core and facesheet solids to an STL file. This function fuses the list of core and facesheet solids into single solids, synchronizes the CAD model, and writes the resulting geometry to a file .

                """
                # Saving CAD geometry with combined solids
                CORE_SOLIDS = CAD.fuse(
                    [(3, CORE_SOLIDS[0])], [(3, tag) for tag in CORE_SOLIDS[1:]]
                )[0][0][1]
                FACESHEET_SOLIDS = CAD.fuse(
                    [(3, FACESHEET_SOLIDS[0])],
                    [(3, tag) for tag in FACESHEET_SOLIDS[1:]],
                )[0][0][1]
                CAD.synchronize()
                gmsh.write(f"file.{run_type}")

            def export_mesh_panel():
                """
                Exports the mesh for a panel model in ABAQUS format.
                Steps performed:
                1. Adds physical groups for facesheet and core solids.
                2. Generates the mesh with options for block elements and second-order elements if specified.
                3. Mirrors the mesh to create the full unit cell using predefined mirror steps.
                4. Tessellates the mesh to create the full panel if multiple repetitions are specified in the X or Y directions.
                5. Sets options for saving groups of elements and nodes.
                6. Writes the mesh to an ABAQUS .inp file.
                """

                # Physical groups
                MODEL.addPhysicalGroup(3, FACESHEET_SOLIDS, name="FACESHEET")
                MODEL.addPhysicalGroup(3, CORE_SOLIDS, name="CORE")
                CAD.synchronize()

                # Mesh
                if self.mesh_params.element_shape == "BLOCK":
                    gmsh.option.setNumber("Mesh.RecombineAll", 1)
                    if self.mesh_params.element_order == 2:
                        gmsh.option.setNumber("Mesh.ElementOrder", 2)
                        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
                    MESH.generate(3)

                # Remove duplicate nodes and renumber
                MESH.removeDuplicateNodes()
                MESH.renumberNodes()
                MESH.renumberElements()
                CAD.synchronize()

                # Mirror steps to make the full unit cell
                mirror_steps = [
                    [1, 1, -1],  # Z
                    [-1, 1, 1],  # X
                    [-1, 1, -1],  # X and Z
                    [1, -1, 1],  # Y
                    [1, -1, -1],  # Y and Z
                    [-1, -1, 1],  # X and Y
                    [-1, -1, -1],  # X, Y and Z
                ]
                tesselate(mirror_steps, None)

                # Pattern to make the full panel
                if self.variables.nLX > 1 or self.variables.nLY > 1:
                    tesselate(None, [self.variables.nLX, self.variables.nLY])

                # Saving ABAQUS geometry
                gmsh.option.setNumber("Mesh.SaveGroupsOfElements", -1000)
                gmsh.option.setNumber("Mesh.SaveGroupsOfNodes", -100)
                gmsh.write("file.inp")

            ###############################################
            # Generate the geometry of the GATOR panel
            ###############################################

            # Add points
            NODES = {}
            for label, value in COORDINATES.items():
                NODES[label] = CAD.addPoint(*value)

            # Add lines and set transfinite curves
            LINES = {}
            for line_set in LINE_CONNECTIVITY:
                line_lengths = [
                    np.linalg.norm(COORDINATES[str(a)] - COORDINATES[str(b)])
                    for a, b in line_set
                ]
                num_nodes = (
                    int(np.ceil(max(line_lengths) / self.mesh_params.mesh_density)) + 1
                )
                for a, b in line_set:
                    line_name = f"{a}_{b}"
                    LINES[line_name] = CAD.addLine(NODES[str(a)], NODES[str(b)])
                    CAD.synchronize()
                    MESH.setTransfiniteCurve(LINES[line_name], num_nodes)

            # Core surfaces
            CORE_SURFACES = [
                build_surface(surface) for surface in SOLID_SURFACE_CONNECTIVITY
            ]

            # Facesheet surfaces
            FACESHEET_SURFACES = []
            if self.variables.facesheet_thickness != 0.0:
                FACESHEET_SURFACES = [
                    build_surface(surface)
                    for surface in FACESHEET_SURFACES_CONNECTIVITY
                ]
                num_layers = int(
                    np.ceil(
                        self.variables.facesheet_thickness
                        / self.mesh_params.mesh_density
                    )
                )
                FACESHEET_SOLIDS = build_solid(
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

            # synchronize CAD after facesheet solid generation
            CAD.synchronize()

            # Collect faces for core extrusion
            CORE_EXTRUDED_FACES = []
            for nodeLabel in CORE_END_SURFACE_BOX:
                min_x = COORDINATES[str(nodeLabel[0][0])][0]
                min_y = COORDINATES[str(nodeLabel[0][1])][1]
                max_x = COORDINATES[str(nodeLabel[1][0])][0]
                max_y = COORDINATES[str(nodeLabel[1][1])][1]
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

            # Extrude solid core
            num_layers = int(
                np.ceil(
                    (self.variables.core_thickness / 2.0)
                    / self.mesh_params.mesh_density
                )
            )
            CORE_SOLIDS = build_solid(
                CORE_EXTRUDED_FACES, self.variables.core_thickness / 2.0, num_layers
            )

            # Synchronize CAD after core solid generation
            CAD.synchronize()

            if run_type in ["stl"]:
                # Combing to parts and saving the CAD file
                export_geometry_unit(CORE_SOLIDS, FACESHEET_SOLIDS, run_type)
            elif run_type == "inp":
                # export mesh panel
                export_mesh_panel()
            else:
                raise Exception(f"Error in run_type: {run_type}")

        ##########################################
        # Run GMSH to great panel geometry
        #########################################

        # Initialise
        gmsh.initialize()

        eps = np.finfo(np.float32).eps  # machine epsilon
        gmsh.option.setNumber(
            "General.Verbosity", 2
        )  # Reduce verbosity, print error and warnings

        # CAD engine (use OCC instead of geo)
        MODEL = gmsh.model
        CAD = MODEL.occ
        MESH = MODEL.mesh

        # Model generation
        generate_geometry_params()
        generate_cad("stl")

        # # Show the model
        # gmsh.fltk.run()

        # Exit gmsh API
        gmsh.finalize()


if __name__ == "__main__":
    design = GATOR(
        Variables(
            core_type="ZPR",
            nLX=2,
            nLY=2,
            theta=np.deg2rad(60.0),
            chevron_wall_length=Units.mm2m(12.5),
            chevron_thickness=Units.mm2m(1.0),
            chevron_separation=Units.mm2m(6.0),
            rib_thickness=Units.mm2m(1.0),
            core_thickness=Units.mm2m(11.0),
            facesheet_thickness=Units.mm2m(0.8),
        ),
        MeshParams(element_type="C3D8R", mesh_density=Units.mm2m(1.0)),
    )

    print(design.variables)
    print(design.derived_variables)
    print(design.mesh_params)
