import pickle
import lzma
import os
import matplotlib.pyplot as plt
import numpy as np
import math

#DATA_FOLDER = "../../../DATA"
DATA_FOLDER = "DATA"

colors = {
  "yellow": "#FBBC05",
  "green": "#34A853",
  "blue": "#4285F4",
  "red": "#EA4335",
  "black": "black",
  "purple": "#410093"
}

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def main():
	timesteps_did_not_finish = 400

	DATAdir = "date_2022_09_20_time_14_09_10_thread_IDs_0_2"  # folder for which simulation data is saved
	mapID = "C_shape"  # ID of the map for which the simulation should be replayed
	
	#DATAdir = "date_2022_09_20_time_08_02_43_thread_IDs_0_2"
	#mapID = "V_shape"  # ID of the map for which the simulation should be replayed

	#DATAdir = "date_2022_09_20_time_19_07_11_thread_IDs_0_2"  # folder for which simulation data is saved
	#mapID = "double_U_shape"  # ID of the map for which the simulation should be replayed

	#mapID = "U_shape"  # ID of the map for which the simulation should be replayed


	DATAdir_ = DATA_FOLDER + "/" + DATAdir + "/" + mapID
	DATAfiles = list_files(DATAdir_)

	pickleFiles = []
	for i in range(len(DATAfiles)):
	    if DATAfiles[i].endswith(".xz"):
	        pickleFiles.append(DATAfiles[i])

	timestepsUsed = []
	travelledDistance = []
	time_used = []

	all_step_distances = []

	if not pickleFiles: # empty list
	    print("File with trajectory not found!")
	    return
	else:
		didNotFinish = 0
		didNotFinish_paths = []
		total = 0
		for pickleFile in pickleFiles:
			total = total + 1
			f = lzma.open(pickleFile, 'rb')
			data = pickle.load(f)

			#meter2pixel = data["meter2pixel"]
			#percentage_explored = data["percentage_explored"]
			timesteps = len(data["explored_map"]["t"])

			time_used.append(sum(data["metrics"]["time_pr_iteration"]))

			dist = 0
			for tau in range(1,len(data["trajectory"])):
				all_step_distances.append(np.sqrt(	(data["trajectory"][tau][0]-data["trajectory"][tau-1][0])**2 +
								 					(data["trajectory"][tau][1]-data["trajectory"][tau-1][1])**2))
				dist = dist + all_step_distances[-1]

			if timesteps < timesteps_did_not_finish:
				timestepsUsed.append(timesteps)
				travelledDistance.append(dist)
			else:
				didNotFinish = didNotFinish + 1
				didNotFinish_paths.append(pickleFile)


			print("Loaded file number " + str(total) + ": "+ str(pickleFile))

	print("N_samples:	" + str(len(pickleFiles)))
	print("Did not finish after " + str(timesteps_did_not_finish) + " timesteps: " + str(didNotFinish))
	print("Timesteps Used mean: " + str(np.mean(timestepsUsed)) + "	(Interval: " + str(np.min(timestepsUsed)) + " - " + str(np.max(timestepsUsed)) + ")")
	print("Average time pr timestep: " + str(np.sum(time_used)/np.sum(timestepsUsed)))
	print("Travelled distance mean: " + str(np.mean(travelledDistance)) + "	(Interval: " + str(np.min(travelledDistance)) + " - " + str(np.max(travelledDistance)) + ")")
	print("Average step length: " + str(np.sum(travelledDistance)/np.sum(timestepsUsed))  + "	(Interval: " + str(np.min(all_step_distances)) + " - " + str(np.max(all_step_distances)) + ")")
	print("Average speed: " + str(np.sum(travelledDistance)/np.sum(time_used)))


	#print("\n")
	print("Did Not Finish:")
	for file in didNotFinish_paths:
		print(file)

	plt.figure()
	plt.hist(timestepsUsed)

	plt.figure()
	plt.hist(travelledDistance)

if __name__ == '__main__':
    main()
    plt.show()