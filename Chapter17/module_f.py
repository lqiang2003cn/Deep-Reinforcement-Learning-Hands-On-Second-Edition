import sys
from modulefinder import ModuleFinder
finder = ModuleFinder()
# Pass the name of the python file of interest
finder.run_script(sys.argv[1])
# This is what's different from @Lila's script
finder.report()