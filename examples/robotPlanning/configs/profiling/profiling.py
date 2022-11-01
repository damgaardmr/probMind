import cProfile
import pstats
from pstats import SortKey

import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
#parentdir = os.path.dirname(parentdir)
os.chdir(parentdir)
sys.path.insert(0, parentdir) 

import main

def loadProfile(filePath, sortKey):
	p = pstats.Stats(filePath)
	#p.strip_dirs().sort_stats(sortKey).print_stats()
	p.strip_dirs().sort_stats(sortKey).print_stats()


if __name__ == '__main__':
	fileName = "profiling_stats"
	filePath = currentdir.replace(parentdir+"/","") + "/" + fileName
	sortKey = "cumtime"
	#sortKey = "filename"


	if False: # profile
		args = ["-cpu_cores", 1, 
				"-config_folder", currentdir]
		cProfile.run("main.main(args)", filePath)

	loadProfile(filePath, sortKey)