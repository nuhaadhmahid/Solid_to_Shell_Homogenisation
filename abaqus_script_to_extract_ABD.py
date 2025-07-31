
# ABAQUS SCIPTING MODULES
from abaqusConstants import *

# ABAQUS ODB MODULES
from odbAccess import *

# OTHER PYTHON PACKAGES
import os  # for file directory operations
import numpy as np
import json # to read the filenames
import sys

import traceback

# EXTRACTION
def data_extraction(case_folder, case_number):
    """
    Extracts the stiffness matrix from the ODB file and saves it in a JSON format.
    """

    # Reference indices for strain/curvature
    strain_ref_idx = {"E11": 0, "E22": 1, "E12": 2, "K11": 3, "K22": 4, "K12": 5}

    # Stiffness matrix terms and indices
    terms = ["A", "B", "D"]
    indices = [11, 12, 16, 22, 26, 66]
    stiffness = {f"{m}{i}": [] for m in terms for i in indices}
    stiffness_idx = {
        "A11": (0, 0), "A12": (0, 1), "A16": (0, 2),
        "A22": (1, 1), "A26": (1, 2), "A66": (2, 2),
        "B11": (0, 3), "B12": (0, 4), "B16": (0, 5),
        "B22": (1, 4), "B26": (1, 5), "B66": (2, 5),
        "D11": (3, 3), "D12": (3, 4), "D16": (3, 5),
        "D22": (4, 4), "D26": (4, 5), "D66": (5, 5)
    }

    odb = None
    try:
        # Load panel data
        panel_path = os.path.join(case_folder, "input", f"{case_number}_panel_data.json")
        with open(panel_path, "r") as f:
            panel_data = json.load(f)
        area = panel_data["lx"] * panel_data["ly"]

        # Open ODB file
        odb_path = os.path.join(case_folder, "odb", f"{case_number}_RVE.odb")
        odb = openOdb(odb_path)

        # Node sets
        instance = odb.rootAssembly.instances['PART-1-1']
        strain_set = instance.nodeSets['STRAIN']
        curvature_set = instance.nodeSets['CURVATURE']

        # Loop over steps
        for step in odb.steps.values():
            if not step.loadCases:
                continue  # Skip non-homogenisation steps

            # Loop over frames
            for frame in step.frames:
                if frame.frameId == 0:
                    continue  # Skip initial frame

                strain_state = frame.description[-3:]
                if strain_state not in strain_ref_idx:
                    continue  # Skip unknown strain states

                rf = frame.fieldOutputs['RF']
                load_strain = rf.getSubset(region=strain_set).values[0].data
                load_curvature = rf.getSubset(region=curvature_set).values[0].data

                abd_col = np.array([
                    load_strain[0] / area,
                    load_strain[1] / area,
                    load_strain[2] / area,
                    load_curvature[0] / area,
                    load_curvature[1] / area,
                    load_curvature[2] / area
                ])
                abd = np.zeros((6, 6))
                abd[:, strain_ref_idx[strain_state]] = abd_col

                # Collect ABD terms
                for term, (i, j) in stiffness_idx.items():
                    stiffness[term].append(abd[i, j])

        # Save results
        data_path = os.path.join(case_folder, "data", f"{case_number}_stiffness.json")
        with open(data_path, 'w') as f:
            json.dump(stiffness, f, indent=4)

    except:
        traceback.print_exc()
    finally:
        if odb is not None:
            odb.close()


if __name__ == '__main__':

    # ARGUMENTS
    case_folder = sys.argv[-2]
    case_number = int(sys.argv[-1])

    try:
        data_extraction(case_folder, case_number)
    except:
        traceback.print_exc()


