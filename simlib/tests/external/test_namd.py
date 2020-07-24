
from simlib.external import namd, NAMDConfiguration
import os
import time

os.chdir('samples/namd-tutorial-files/1-3-box/')

# engine = NAMD(config_file='ubq_wb_eq.conf', executable='namd2.exe', background=True)
# engine.start()
#
# k = 0
# while engine.poll() is None:
#     k += 1
#     print(k)
#     time.sleep(10)


namd_config = NAMDConfiguration('ubq_wb_eq.conf')
print(namd_config)
