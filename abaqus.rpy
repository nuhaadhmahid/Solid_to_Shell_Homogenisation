# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 2023 replay file
# Internal Version: 2022_09_28-19.11.55 183150
# Run by nm16829 on Fri Jul 18 16:39:05 2025
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(1.3724, 1.37037), width=202.017, 
    height=135.941)
session.viewports['Viewport: 1'].makeCurrent()
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
execfile('extract_ABD.py', __main__.__dict__)
#* SyntaxError: ('invalid syntax', ('extract_ABD.py', 49, 76, '        odb_file 
#* = os.path.join(case_folder, "odb", f"{case_number}_RVE.odb")\n'))
