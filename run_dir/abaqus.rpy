# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023 replay file
# Internal Version: 2022_09_28-19.11.55 183150
# Run by nm16829 on Thu Jun 19 12:37:16 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=316.748931884766, 
    height=204.733337402344)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.ModelFromInputFile(name='file', 
    inputFileName='C:/Users/nm16829/Documents/PostDoc_Local_Drive/Solid_to_Shell_Homogenisation/file.inp')
#: The model "file" has been created.
#: The part "PART-1" has been imported from the input file.
#: The model "file" has been imported from an input file. 
#: Please scroll up to check for error and warning messages.
session.viewports['Viewport: 1'].assemblyDisplay.setValues(
    optimizationTasks=OFF, geometricRestrictions=OFF, stopConditions=OFF)
a = mdb.models['file'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
del mdb.models['file']
a = mdb.models['Model-1'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
mdb.ModelFromInputFile(name='file', 
    inputFileName='C:/Users/nm16829/Documents/PostDoc_Local_Drive/Solid_to_Shell_Homogenisation/file.inp')
#: The model "file" has been created.
#: The part "PART-1" has been imported from the input file.
#: The model "file" has been imported from an input file. 
#: Please scroll up to check for error and warning messages.
a = mdb.models['file'].rootAssembly
session.viewports['Viewport: 1'].setValues(displayedObject=a)
