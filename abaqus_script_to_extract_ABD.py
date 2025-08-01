
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

# EXTACTION
def data_extraction(case_folder, case_number):
    """
    Extracts the stiffness matrix from the ODB file and saves it in a JSON format.
    """

    # reference index
    Strain_Reference_Index = {"E11":0,"E22":1,"E12":2,"K11":3,"K22":4,"K12":5}

    # collector
    STIFFNESS={"%s%i"%(matrix,term):[] for matrix in ["A", "B", "D"] for term in [11, 12, 16, 22, 26, 66]}
    STIFFNESS_INDEX= {
        "A11":[0,0], "A12":[0,1], "A16":[0,2],
                     "A22":[1,1], "A26":[1,2],
                                  "A66":[2,2],
        "B11":[0,3], "B12":[0,4], "B16":[0,5],
                     "B22":[1,4], "B26":[1,5],
                                  "B66":[2,5],
        "D11":[3,3], "D12":[3,4], "D16":[3,5],
                     "D22":[4,4], "D26":[4,5],
                                  "D66":[5,5]
    }
    
    try:
        # READ DEFAULT INPUTS
        with open(os.path.join(case_folder, "input", "%i_panel_data.json"%(case_number)), "r") as json_file:
            panel_data = json.load(json_file)

        # reference area
        Area = panel_data["lx"]*panel_data["ly"]


        # ODB Files
        odb_file = os.path.join(case_folder, "odb", "%i_RVE.odb"%(case_number))
        odb=openOdb(odb_file)

        # Sets
        STRAIN_set = odb.rootAssembly.instances['PART-1-1'].nodeSets['STRAIN']
        CURVATURE_set = odb.rootAssembly.instances['PART-1-1'].nodeSets['CURVATURE']

        # Step
        steps=odb.steps
        for StepName, step in steps.items():

            if bool(step.loadCases): Homogenisation = True # print("Homogenisation Step")  
            else: Homogenisation = False # print("General Step")

            if Homogenisation:
                # Frames
                frames=step.frames # you could pick whatever frame you want here

                # Homogenisation Loads
                ABD = np.zeros((6,6))
                for frame in frames:

                    # loading mode for homoginisation
                    if frame.frameId==0: continue
                    else:Strain_State = frame.description[-3:]

                    # Loading nodes' reaction load
                    reactionForce = frame.fieldOutputs['RF']
                    LOAD_on_STRAIN_node = reactionForce.getSubset(region=STRAIN_set).values[0].data
                    LOAD_on_CURVATURE_node = reactionForce.getSubset(region=CURVATURE_set).values[0].data

                    # ABD from load data, assuming applied strain is exx=exy=rxy=1 
                    ABD_column = np.array([
                        LOAD_on_STRAIN_node[0]/Area,
                        LOAD_on_STRAIN_node[1]/Area,
                        LOAD_on_STRAIN_node[2]/Area,
                        LOAD_on_CURVATURE_node[0]/Area,
                        LOAD_on_CURVATURE_node[1]/Area,
                        LOAD_on_CURVATURE_node[2]/Area
                    ])
                    ABD[:,Strain_Reference_Index[Strain_State]] = ABD_column

                # collecting the ABD matrix
                for term in STIFFNESS.keys():
                    STIFFNESS[term].append(ABD[STIFFNESS_INDEX[term][0],STIFFNESS_INDEX[term][1]])

        # save data
        data_file = os.path.join(case_folder, "data", "%i_stiffness.json"%(case_number))
        with open(data_file, 'w') as json_file:
            json.dump(STIFFNESS, json_file, indent = 4)

        if 'odb' in locals():
            odb.close()
    
    except Exception as e:
        print(e)
        traceback.print_exc()

        if 'odb' in locals():
            odb.close()


if __name__ == '__main__':

    # ARGUMENTS
    case_folder = sys.argv[-2]
    case_number = int(sys.argv[-1])

    try:
        data_extraction(case_folder, case_number)
    except Exception as e:
        print(e)
        traceback.print_exc()


