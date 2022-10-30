import os
import sys
import inspect
import copy
from pathlib import Path
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import pickle
import lzma

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
os.chdir(parentdir)
sys.path.insert(0, parentdir) 

import main

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def all_subdirs_of(b='.'):
  result = []
  for d in os.listdir(b):
    bd = os.path.join(b, d)
    if os.path.isdir(bd): result.append(bd)
  return result

def get_initial_pose(map_IDs_file, startPoseFilePath):
	latest_subdir = max(all_subdirs_of(b="DATA"), key=os.path.getmtime)

	initial_poses = {}

	print("Going to load old data...")
	with open(map_IDs_file,'r') as mapIDfile:
		for mapID in mapIDfile:
			dir_ = latest_subdir + "/" + mapID.replace("\n","")
			files = list_files(dir_)

			for file in files:
				if mapID.replace("\n","") in file and file.endswith(".xz"):
					pickleFile = file
			print(pickleFile)
			f = lzma.open(pickleFile, 'rb')
			data = pickle.load(f)

			# print("data[config_poseInit]: " + str(data["config_poseInit"]))

			initial_poses[mapID.replace("\n","")] = data["config_poseInit"]

	print("Saving initial positions...")
	with lzma.open(Path(startPoseFilePath), "wb") as f:
		pickle.dump(initial_poses, f)




if __name__ == '__main__':

	#DELETE_OLD_INITIAL_POSE_FILE = True
	DELETE_OLD_INITIAL_POSE_FILE = False

	currentFolder = str(currentdir.split("configs")[1])
	experiment_folder = "configs"+currentFolder

	map_id_file = experiment_folder + '/map_ids.txt'
	#map_id_file = experiment_folder + '/map_ids_remaining.txt'
	#map_id_file = experiment_folder + '/map_ids_test.txt'
	#map_id_file = experiment_folder + '/map_ids_test2.txt'


	agents = 	{"damgaard2022AKS":		{"useStartPosesFromLastExperiment": False,
										 "base_configs_file": experiment_folder.replace("damgaard2022RGS/experiment2","damgaard2022AKS").replace("damgaard2022RGS\experiment2","damgaard2022AKS") + "/configs.yaml",
										 "cpu_cores": 6
										},
			 	"damgaard2022RGS":		{"useStartPosesFromLastExperiment": True,
			 							 #"useStartPosesFromLastExperiment": False,
			 							 "base_configs_file": experiment_folder.replace("experiment2","experiment1") + "/base_configs.yaml",
			 							 "cpu_cores": 1
										},
			 	}

	startPoseFilePath = experiment_folder + "/" + "startPoses" + ".xz"
	if DELETE_OLD_INITIAL_POSE_FILE:
		if os.path.exists(startPoseFilePath):
			os.remove(startPoseFilePath)


	#for agent in agents.keys():
	for agent in ["damgaard2022RGS"]:
		#for agent in ["damgaard2022AKS"]:

		config_folder = experiment_folder + "/configs"
		# create folders to save data
		if not os.path.exists(config_folder):
				os.mkdir(config_folder)

		config_folder = config_folder + "/" + agent
		# create folders to save data
		if not os.path.exists(config_folder):
			os.mkdir(config_folder)

		# copy file with map IDS
		map_id_file_ = config_folder + "/map_ids.txt"
		with open(map_id_file,'r') as firstfile, open(map_id_file_,'w') as secondfile:
			# read content from first file
			for line in firstfile:
				# append content to second file
				secondfile.write(line)

		config_file_in = agents[agent]["base_configs_file"]
		print(agent)
		print(experiment_folder)
		print(experiment_folder.replace("damgaard2022RGS/experiment2","damgaard2022AKS").replace("damgaard2022RGS\experiment2","damgaard2022AKS") + "/configs.yaml")
		print(config_file_in)
		with open(Path(config_file_in), 'r') as stream:
			try:
				configs = yaml.load(stream, Loader=Loader)
			except yaml.YAMLError as exc:
				print(exc)

		if agents[agent]["useStartPosesFromLastExperiment"]:
			if not os.path.exists(startPoseFilePath):
				get_initial_pose(map_id_file, startPoseFilePath)
			configs["resetRandomPose"] = 0
			configs["startPose"] = startPoseFilePath
		else:
			configs["resetRandomPose"] = 1

		for key in agents[agent].keys():
			configs[key] = agents[agent][key]

		configs["json_dir"] = "../../HouseExpo/json/"
		configs["t_max"] = 200
		configs["T_delta"] = 2
		configs["imageScaling"] = 3
		configs["reachGoalMode"] = False
		configs["lidar"]["range"] = 2 # like in the damgaard2022AKS experiment
		configs["robotRadius"] = 0.2 # 0.2 was used in the original damgaard2022AKS experiment
		
		# Used for RGS_small
		#configs["initial_3_sigma"] = 0.1 # 0.2 was used in the original damgaard2022AKS experiment
		#configs["PlanningParams"]["movement_3_sigma"] = 0.1 # 0.2 was used in the original damgaard2022AKS experiment
		#configs["robotRadius"] = 0.1 # 0.2 was used in the original damgaard2022AKS experiment


		config_file_out = config_folder + "/configs.yaml"
		with open(Path(config_file_out), 'w') as stream:
			try:
				yaml.dump(configs, stream, default_flow_style=False, Dumper=Dumper)
			except yaml.YAMLError as exc:
				print(exc)

		config_folder = experiment_folder + "/configs/" + agent
		args = ["-config_folder", config_folder, 
				"-cpu_cores", agents[agent]["cpu_cores"],
				"-total_cluster_threads", agents[agent]["cpu_cores"]]
		main.main(args)