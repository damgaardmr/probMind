from pathlib import Path
import os
from datetime import datetime, timedelta

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import sim

configs_folder = "configs/damgaard2022SVIFDPR/Antipodal_Circle_Swapping/"

N_Robots = [2, 4, 8, 16, 32]  # 32 causes collisions from the beginning with random initialization
real_time_factor = [0.9, 0.8, 0.5, 0.15, 0.05]  # for simulation on a PC with an intel i5-12600k CPU
max_expected_sim_times = [45, 50, 70, 80, 100]

#N_Robots = [32, 16, 8, 4, 2]  # 32 causes collisions from the beginning with random initialization
#real_time_factor = [0.05, 0.15, 0.5, 0.8, 0.9]  # for simulation on a PC with an intel i5-12600k CPU
#max_expected_sim_times = [90, 70, 65, 60, 45]

#N_Robots = [2]  
#real_time_factor = [0.9]
#N_Robots = [4]  
#real_time_factor = [0.8]
#N_Robots = [8]  
#real_time_factor = [0.5]
#N_Robots = [16]  
#real_time_factor = [0.15]
#N_Robots = [32]  # 32 causes collisions from the beginning with random initialization
#real_time_factor = [0.05]  # for simulation on a PC with an intel i5-12600k CPU
#max_expected_sim_times = [100]

def main():
	now = datetime.now()
	date = now.strftime("%Y_%m_%d")
	current_time = now.strftime("%H_%M_%S")
	DataDir = "DATA/date_" + date + "_time_" + current_time

	with open(Path(configs_folder + "configs.yaml"), 'r') as stream:
		try:
			configs = yaml.load(stream, Loader=Loader)
		except yaml.YAMLError as exc:
			print(exc)

	expected_duration = 0
	for sim_ID in range(len(N_Robots)):
		expected_duration = expected_duration + configs["N_simulations"] * (max_expected_sim_times[sim_ID] / real_time_factor[sim_ID])

	expected_duration = timedelta(seconds=expected_duration)
	print("Expected Duration: " + str(expected_duration))
	Expected_Finish = now + expected_duration
	print("Expected Finish: " + Expected_Finish.strftime("%H:%M:%S"))

	for sim_ID in range(len(N_Robots)):
		# create config file...
		configs["N_Robots"] = N_Robots[sim_ID]
		configs["real_time_factor"] = real_time_factor[sim_ID]
		with open(Path(configs_folder + "configs_" + str(N_Robots[sim_ID]) + ".yaml"), 'w') as stream:
			try:
				yaml.dump(configs, stream, default_flow_style=False, Dumper=Dumper)
			except yaml.YAMLError as exc:
				print(exc)

		# run simulations for sim_ID
		args = ["-config_file", configs_folder + "configs_" + str(N_Robots[sim_ID]) + ".yaml",
				"-data_dir", DataDir + "/" + str(N_Robots[sim_ID])]
		sim.main(args)


if __name__ == '__main__':
    main()