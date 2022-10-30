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

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
parentdir = os.path.dirname(parentdir)
os.chdir(parentdir)
sys.path.insert(0, parentdir) 

import main

if __name__ == '__main__':
	# consider with the U_shape and cluttered_map
	# consider with and without knowledge about the goal location

	N_samples_pr_map = 100
	N_samples_pr_map = 1
	cpu_cores = 1
	experiment_folder = "configs"+str(currentdir.split("configs")[1])
	map_settings = {#"U_shape": 			{"startPose": 		[-6.0, 0.0, 0.0], 
					#					 "startPose": 		[-2, 0,0],
					#					 "goalPosition": 	[5.5, 3.5],
					#					},
					"V_shape":			{"startPose": 		[-9, 1.0, 0.0],
										 "goalPosition": 	[15.0, 7.5],
										 "t_max":			400,
										 },
					"C_shape":			{"startPose": 		[-8.0, -5.0, 0.0],
										 "goalPosition": 	[16.75, 7.5],
										 "t_max":			400,
										},
					"double_U_shape":	{"startPose": 		[-1.75, 4, 0],
										"goalPosition": 	[11.5, 12.5],
										"t_max":			400,
										},
					}

	#for map_ in ["C_shape"]:
	for map_ in map_settings.keys():
		config_folder = experiment_folder + "/configs"
		# create folders to save data
		if not os.path.exists(config_folder):
			os.mkdir(config_folder)

		config_folder = config_folder + "/" + map_
		# create folders to save data
		if not os.path.exists(config_folder):
			os.mkdir(config_folder)

		map_id_file = config_folder + "/map_ids.txt"
		with open(map_id_file, 'w') as f:
			for n in range(N_samples_pr_map):
				if n < N_samples_pr_map-1:
					f.write(map_ + "\n")
				else:
					f.write(map_)

		config_file_in = experiment_folder + "/base_configs.yaml"
		with open(Path(config_file_in), 'r') as stream:
			try:
				configs = yaml.load(stream, Loader=Loader)
			except yaml.YAMLError as exc:
				print(exc)

		configs["t_max"] = map_settings[map_]["t_max"]
		configs["startPose"]["x"] = map_settings[map_]["startPose"][0]
		configs["startPose"]["y"] = map_settings[map_]["startPose"][1]
		configs["startPose"]["theta"] = map_settings[map_]["startPose"][2]
		configs["goalPosition"]["x"] = map_settings[map_]["goalPosition"][0]
		configs["goalPosition"]["y"] = map_settings[map_]["goalPosition"][1]
		configs["json_dir"] = configs["json_dir"].replace("configs", experiment_folder+"/maps/")


		config_file_out = config_folder + "/configs.yaml"
		with open(Path(config_file_out), 'w') as stream:
			try:
				yaml.dump(configs, stream, default_flow_style=False, Dumper=Dumper)
			except yaml.YAMLError as exc:
				print(exc)

		config_folder = experiment_folder + "/configs/" + map_
		args = ["-config_folder", config_folder, 
				"-cpu_cores", cpu_cores,
				"-total_cluster_threads", cpu_cores]
		main.main(args)