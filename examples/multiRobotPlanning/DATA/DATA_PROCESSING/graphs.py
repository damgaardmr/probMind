import pickle
import lzma
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import imageio
import tikzplotlib
from pathlib import Path



# latex formatting strings:
# https://matplotlib.org/stable/tutorials/text/usetex.html
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
# remember that special charathers in strings in figures have to have prober escaping: i.e. use "\%" instead of "%"
plt.rcParams['text.usetex'] = True

colors = {
  "yellow": "#FBBC05",
  "green": "#34A853",
  "blue": "#4285F4",
  "red": "#EA4335",
  "black": "black",
  "purple": "#410093"
}


def main():
	sim_id_to_exclude = []
	#LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle/LOGS"
	#N_simulations = 50

	#LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle_Swapping/SVMMN/2/LOGS"
	#LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle_Swapping/SVMMN/4/LOGS"
	#LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle_Swapping/SVMMN/8/LOGS"
	#LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle_Swapping/SVMMN/16/LOGS"
	LOGSdir = "../damgaard2022SVIFDPR/Antipodal_Circle_Swapping/SVMMN/32/LOGS"
	N_simulations = 10

	sim_time_all_sims = []
	dists_true_min_all_sims = []
	N_msgs_received_all_sims = []
	N_times_goal_reached_all_sims = []
	delta_T_between_goals_all_sims_first = []
	delta_dist_all_sims_first = []
	delta_T_between_goals_all_sims = []
	delta_dist_all_sims = []

	actions_trans_all_sims = []
	actions_rot_all_sims = []

	sim_ids = list(range(1, N_simulations+1))
	for i in range(len(sim_id_to_exclude)):
		sim_ids.remove(sim_id_to_exclude[i])

	for sim_id in sim_ids:
		print("Loading data for simulation number " + str(sim_id))
		LOGfile = LOGSdir + "/log_sim_" + str(sim_id) + ".xz"

		f = lzma.open(LOGfile, 'rb')
		log_dict = pickle.load(f)

		N_robots = len(log_dict["robot_poses"][0])
		poses = log_dict["robot_poses"]
		poses_est = log_dict["robot_poses_est"]
		sim_time = log_dict["sim_time"]
		r_robots = log_dict["r_robots"]
		N_msgs_received = log_dict["N_msgs_received"]
		current_goals = log_dict["current_goals"]
		if "acts" in log_dict:
			acts = log_dict["acts"]

		sim_time_all_sims.append(sim_time)

		# # ########################### Seperating distance calc ###########################
		dists_true = []
		dists_true_min = []
		dists_est = []
		dists_est_min = []
		for tau in range(len(sim_time)):
			dists_true_ = []
			dists_est_ = []
			for n in range(N_robots):
				for m in range(n+1, N_robots):	
					dist_vector = np.array([poses[tau][n][0] - poses[tau][m][0],
											poses[tau][n][1] - poses[tau][m][1]])
					dist_true = np.linalg.norm(dist_vector) - r_robots[n] - r_robots[m]
					if dist_true <= 0.0:
						print("'dist_true <= 0.0' in sim " + str(sim_id) + " at tau " + str(tau) + " s for robots with ID " + str(n) + " and " + str(m))
					dists_true_.append(dist_true)
					
					dist_vector = np.array([poses_est[tau][n][0] - poses_est[tau][m][0],
											poses_est[tau][n][1] - poses_est[tau][m][1]])
					dist_est = np.linalg.norm(dist_vector) - r_robots[n] - r_robots[m]
					dists_est_.append(dist_est)
					#if dist_est <= 0.0:
					#	print("'dist_est <= 0.0' in sim " + str(sim_id) + " at tau " + str(tau) + " s for robots with ID " + str(n) + " and " + str(m))
			dists_true.append(dists_true_)
			dists_true_min.append(np.min(dists_true_))
			dists_est.append(dists_est_)
			dists_est_min.append(np.min(dists_est_))

		dists_true_min_all_sims.append(dists_true_min)

		# ########################### actions ###########################
		#	actions_trans_all_sims = []
		# actions_rot_all_sims = []
		if "acts" in log_dict:
			for i in range(len(acts)):
				for n in range(N_robots):
					actions_trans_all_sims.append(acts[i][n][0])
					actions_rot_all_sims.append(acts[i][n][1])

		# ########################### Msg received calc ###########################
		N_msgs_received_all_robots = sum(N_msgs_received, [])
		N_msgs_received_all_sims.append(N_msgs_received_all_robots)

		# ########################### goal reached calc ###########################
		current_goals_ = [[] for _ in range(N_robots)]
		for tau in range(len(current_goals)):
			for n in range(N_robots):
				current_goals_[n].append(current_goals[tau][n])

		N_times_goal_reached = []
		current_goal_changed = []
		for n in range(N_robots):
			current_goal_changed.append(np.logical_xor(current_goals_[n][1:],current_goals_[n][:-1]))
			N_times_goal_reached.append(current_goal_changed[n].sum())

		N_times_goal_reached_all_sims.append(N_times_goal_reached)

		# ########################### avg goal travel dist calc ###########################
		np.set_printoptions(threshold=np.inf)
		#print(current_goal_changed[n])
		N_times_goal_reached_list = np.zeros_like(current_goal_changed, dtype=int)
		for n in range(N_robots):
			ctn = 0
			for tau in range(len(current_goals)-1):
				if current_goal_changed[n][tau]:
					ctn = ctn + 1
				N_times_goal_reached_list[n][tau] = ctn



		sim_time_np = np.array(sim_time[0:-1])
		poses_np = np.array(poses[0:-1])

		for n in range(N_robots):
			positions = np.empty([len(sim_time_np),2])
			for tau in range(len(sim_time_np)):
				positions[tau,0] = poses[tau][n][0]
				positions[tau,1] = poses[tau][n][1]

			for i in range(0, N_times_goal_reached[n]):  # other times
				mask = N_times_goal_reached_list[n] == i
				mask_np = np.array(mask, dtype=bool)
				times_tmp = sim_time_np[mask_np]
				delta_T_between_goals = times_tmp[-1] - times_tmp[0]  # time it took to drive between two goals

				positions_masked = positions[mask]
				delta_dist_between_goal = 0   # distance droven between two goals
				for tau in range(1,len(positions_masked)):
					dist_vector = np.array([positions_masked[tau][0] - positions_masked[tau-1][0],
																	positions_masked[tau][1] - positions_masked[tau-1][1]])
					delta_dist_between_goal = delta_dist_between_goal + np.linalg.norm(dist_vector)

				if i == 0:  # first time is a special case due to the artificial increase in potential collisions
					delta_T_between_goals_all_sims_first.append(delta_T_between_goals)
					delta_dist_all_sims_first.append(delta_dist_between_goal)
				else:
					delta_T_between_goals_all_sims.append(delta_T_between_goals)
					delta_dist_all_sims.append(delta_dist_between_goal)


	# # ########################### Seperating distance plot ###########################
	minimum_seperating_distance_pr_sim = []
	for sim_id in range(len(sim_time_all_sims)):
		minimum_seperating_distance_pr_sim.append(np.min(dists_true_min_all_sims[sim_id]))

	#print(minimum_seperating_distance_pr_sim)
	minimum_ = np.min(minimum_seperating_distance_pr_sim)
	MSD_mean = np.mean(minimum_seperating_distance_pr_sim)
	MSD_std = np.std(minimum_seperating_distance_pr_sim)
	MSD_var = np.var(minimum_seperating_distance_pr_sim)

	print("Minimum Seperating Distance pr sim mean: " + str(MSD_mean) + " +- " + str(MSD_std) + " std. (var: " + str(MSD_var) + ")")
	print("minimum of minimums ... : " + str(minimum_))
	fig, ax = plt.subplots(figsize=[6.4, 6.4/2])
	for sim_id in range(len(sim_time_all_sims)):
		if sim_id == 0:
			plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["blue"], label="MSD")
		else:
			plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["blue"])

	# plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["black"], linewidth=3, label="Seperation Distance")
	plt.plot([sim_time[0], sim_time[-1]], [0, 0], linewidth=3, color=colors["black"], label="Collision")
	plt.plot([sim_time[0], sim_time[-1]], [MSD_mean, MSD_mean], linewidth=2, linestyle="dotted", color=colors["black"], label="MSD Mean")
	plt.plot([sim_time[0], sim_time[-1]], [minimum_, minimum_], linewidth=2, linestyle="--", color=colors["black"], label="MSD Minimum")

	plt.legend(loc='upper right')
	plt.xlabel("Time [s]")
	plt.ylabel("Distance [m]")

	tikzFolder = LOGSdir
	tikz_file_path = tikzFolder + "/" + "tikz_Seperating_distance.tikz"
	tikzplotlib.save(tikz_file_path)
	# modify generated tikz file
	width_str = "\\linewidth"
	plot_width = 1.0  # *width_str
	plot_height = 0.3  # *width_str
	fin = open(tikz_file_path, "rt")
	fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
	for line in fin:
	    if "\\begin{axis}[" in line:
	        fout.write(line)
	        fout.write("width={width}{width_str},\n".format(width=plot_width, width_str=width_str))
	        fout.write("height={height}{width_str},\n".format(height=plot_height,width_str=width_str))
	        fout.write("clip marker paths=true,\n")  # fixes error with the order of plotting
	        fout.write("scaled x ticks=false,\n")
	    else:
	        fout.write(line)
	fin.close()
	fout.close()
	os.remove(tikz_file_path)
	os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)

	# # ########################### Msg received histogram ###########################
	N_msgs_received_all_sims = sum(N_msgs_received_all_sims, [])
	bins = np.arange(0, np.max(N_msgs_received_all_sims) + 1.5) - 0.5
	fig, ax = plt.subplots(figsize=[6.4, 6.4/2])
	_ = ax.hist(N_msgs_received_all_sims, bins, color=colors["blue"])
	ax.set_xticks(bins + 0.5)
	plt.xlabel("Messages Received [1]")
	plt.ylabel("Count [1]")

	tikzFolder = LOGSdir
	tikz_file_path = tikzFolder + "/" + "tikz_msg_received_histogram.tikz"
	tikzplotlib.save(tikz_file_path)
	# modify generated tikz file
	width_str = "\\linewidth"
	plot_width = 1.0  # *width_str
	plot_height = 0.64  # *width_str
	fin = open(tikz_file_path, "rt")
	fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
	for line in fin:
	    if "\\begin{axis}[" in line:
	        fout.write(line)
	        fout.write("width={width}{width_str},\n".format(width=plot_width, width_str=width_str))
	        fout.write("height={height}{width_str},\n".format(height=plot_height,width_str=width_str))
	        fout.write("clip marker paths=true,\n")  # fixes error with the order of plotting
	        fout.write("scaled x ticks=false,\n")
	    else:
	        fout.write(line)
	fin.close()
	fout.close()
	os.remove(tikz_file_path)
	os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)


	# ########################### goal reached histogram ###########################
	N_times_goal_reached_all_sims = sum(N_times_goal_reached_all_sims, [])
	bins = np.arange(0, np.max(N_times_goal_reached_all_sims) + 1.5) - 0.5
	fig, ax = plt.subplots(figsize=[6.4, 6.4*0.4])
	_ = ax.hist(N_times_goal_reached_all_sims, bins, color=colors["blue"])
	ax.set_xticks(bins + 0.5)
	plt.xlabel("Number of Times a Goal was Reached [1]")
	plt.ylabel("Count [1]")

	tikzFolder = LOGSdir
	tikz_file_path = tikzFolder + "/" + "tikz_goals_reached_histogram.tikz"
	tikzplotlib.save(tikz_file_path)
	# modify generated tikz file
	width_str = "\\linewidth"
	plot_width = 1.0  # *width_str
	plot_height = 0.3  # *width_str
	fin = open(tikz_file_path, "rt")
	fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
	for line in fin:
	    if "\\begin{axis}[" in line:
	        fout.write(line)
	        fout.write("width={width}{width_str},\n".format(width=plot_width, width_str=width_str))
	        fout.write("height={height}{width_str},\n".format(height=plot_height,width_str=width_str))
	        fout.write("clip marker paths=true,\n")  # fixes error with the order of plotting
	        fout.write("scaled x ticks=false,\n")
	    else:
	        fout.write(line)
	fin.close()
	fout.close()
	os.remove(tikz_file_path)
	os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)

	# ########################### actions ###########################
	if "acts" in log_dict:
		speed_max = 0.22 # m/s
		#	actions_trans_all_sims = []
		# actions_rot_all_sims = []
		actions_trans_all_sims_norm = []
		for i in range(len(actions_trans_all_sims)):
			actions_trans_all_sims_norm.append(actions_trans_all_sims[i]/speed_max)
		fig, ax = plt.subplots(figsize=[6.4, 6.4*0.4])
		_ = ax.hist(actions_trans_all_sims_norm)
		plt.xlabel("Trans Actions [m/s]")
		plt.ylabel("Count [1]")
		action_min = np.min(actions_trans_all_sims)
		action_max = np.max(actions_trans_all_sims)
		action_mean = np.mean(actions_trans_all_sims)
		action_std = np.std(actions_trans_all_sims)
		action_var = np.var(actions_trans_all_sims)
		print("Actions:\n Minimum " + str(action_min) + "	Maximum: " + str(action_max) + "	mean: " + str(action_mean) + "	std: " + str(action_std) + "	variance: " + str(action_var))
		print("fractions of max. Maximum " + str(action_max/speed_max) + " 	mean:  " + str(action_mean/speed_max))

	# ########################### distance between goals ###########################
	if N_simulations == 50:
		R_env = 2
		goal_radius = 0.25
	else:
		R_env = 4
		goal_radius = 0.1
	minimum_distance_first = 2*R_env - goal_radius
	minimum_distance = 2*R_env - 2*goal_radius
	print("\n")
	print("minimum_distance: " + str(minimum_distance))

	minimum_ = np.min(delta_dist_all_sims_first)
	maximum_ = np.max(delta_dist_all_sims_first)
	delta_dist_mean_first = np.mean(delta_dist_all_sims_first)
	delta_dist_std_first = np.std(delta_dist_all_sims_first)
	delta_dist_var_first = np.var(delta_dist_all_sims_first)
	delta_dist_mean_norm_first = delta_dist_mean_first/minimum_distance_first
	delta_dist_std_norm_first = delta_dist_std_first/minimum_distance_first
	delta_dist_var_norm_first = delta_dist_var_first/minimum_distance_first
	print("Delta distance droven before first goal is reach.\n Minimum " + str(minimum_) + "	Maximum: " + str(maximum_) + "	mean: " + str(delta_dist_mean_first) + "	std: " + str(delta_dist_std_first) + "	variance: " + str(delta_dist_var_first))
	print("Normalized Delta distance droven before first goal is reach.\n mean: " + str(delta_dist_mean_norm_first) + "	std: " + str(delta_dist_std_norm_first) + "	variance: " + str(delta_dist_var_norm_first))
	print("\n")

	if N_simulations == 50:
		minimum_ = np.min(delta_dist_all_sims)
		maximum_ = np.max(delta_dist_all_sims)
		delta_dist_mean = np.mean(delta_dist_all_sims)
		delta_dist_std = np.std(delta_dist_all_sims)
		delta_dist_var = np.var(delta_dist_all_sims)
		delta_dist_mean_norm = delta_dist_mean/minimum_distance
		delta_dist_std_norm = delta_dist_std/minimum_distance
		delta_dist_var_norm = delta_dist_var/minimum_distance
		print("Delta distance droven between goals.\n Minimum " + str(minimum_) + "	Maximum: " + str(maximum_) + "	mean: " + str(delta_dist_mean) + "	std: " + str(delta_dist_std) + "	variance: " + str(delta_dist_var))
		print("Normalized Delta distance droven between goals.\n mean: " + str(delta_dist_mean_norm) + "	std: " + str(delta_dist_std_norm) + "	variance: " + str(delta_dist_var_norm))
		print("\n")

	# ########################### time between goals ###########################
	speed_max = 0.22 # m/s
	minimum_delta_T_first = minimum_distance_first / speed_max
	minimum_delta_T = minimum_distance / speed_max
	print("minimum_delta_T: " + str(minimum_delta_T))

	minimum_ = np.min(delta_T_between_goals_all_sims_first)
	maximum_ = np.max(delta_T_between_goals_all_sims_first)
	delta_T_mean_first = np.mean(delta_T_between_goals_all_sims_first)
	delta_T_std_first = np.std(delta_T_between_goals_all_sims_first)
	delta_T_var_first = np.var(delta_T_between_goals_all_sims_first)
	delta_T_mean_norm_first = delta_T_mean_first/minimum_delta_T_first
	delta_T_std_norm_first = delta_T_std_first/minimum_delta_T_first
	delta_T_var_norm_first = delta_T_var_first/minimum_delta_T_first
	print("Delta time before first goal is reach.\n Minimum " + str(minimum_) + "	Maximum: " + str(maximum_) + "	mean: " + str(delta_T_mean_first) + "	std: " + str(delta_T_std_first) + "	variance: " + str(delta_T_var_first))
	print("Normalized Delta time before first goal is reach.\n mean: " + str(delta_T_mean_norm_first) + "	std: " + str(delta_T_std_norm_first) + "	variance: " + str(delta_T_var_norm_first))
	print("\n")

	if N_simulations == 50:
		minimum_ = np.min(delta_T_between_goals_all_sims)
		maximum_ = np.max(delta_T_between_goals_all_sims)
		delta_T_mean = np.mean(delta_T_between_goals_all_sims)
		delta_T_std = np.std(delta_T_between_goals_all_sims)
		delta_T_var = np.var(delta_T_between_goals_all_sims)
		delta_T_mean_norm = delta_T_mean/minimum_delta_T
		delta_T_std_norm = delta_T_std/minimum_delta_T
		delta_T_var_norm = delta_T_var/minimum_delta_T
		print("Delta time between goals.\n Minimum " + str(minimum_) + "	Maximum: " + str(maximum_) + "	mean: " + str(delta_T_mean) + "	std: " + str(delta_T_std) + "	variance: " + str(delta_T_var))
		print("Normalized Delta time between goals.\n mean: " + str(delta_T_mean_norm) + "	std: " + str(delta_T_std_norm) + "	variance: " + str(delta_T_var_norm))




	# ########################### save data... ###########################
	DATA = {}
	DATA["MinimumDistance"] = {}
	DATA["MinimumDistance"]["mean"] = None  # does not make sence since robots can have different dimensions
	DATA["MinimumDistance"]["std"] = None  # does not make sence since robots can have different dimensions
	DATA["MinimumDistance"]["var"] = None  # does not make sence since robots can have different dimensions
	DATA["travelledDistance"] = {}
	DATA["travelledDistance"]["mean"] = delta_dist_mean_first
	DATA["travelledDistance"]["std"] = delta_dist_std_first
	DATA["travelledDistance"]["var"] = delta_dist_var_first
	DATA["travelledTime"] = {}
	DATA["travelledTime"]["mean"] = delta_T_mean_first
	DATA["travelledTime"]["std"] = delta_T_std_first
	DATA["travelledTime"]["var"] = delta_T_var_first

	DATA["MSD"] = {}  # smallest separating distance 
	DATA["MSD"]["mean"] = MSD_mean
	DATA["MSD"]["std"] = MSD_std
	DATA["MSD"]["var"] = MSD_var
	DATA["travelledDistanceNormalized"] = {}
	DATA["travelledDistanceNormalized"]["mean"] = delta_dist_mean_norm_first
	DATA["travelledDistanceNormalized"]["std"] = delta_dist_std_norm_first 
	DATA["travelledDistanceNormalized"]["var"] = delta_dist_var_norm_first
	DATA["travelledTimeNormalized"] = {}
	DATA["travelledTimeNormalized"]["mean"] = delta_T_mean_norm_first
	DATA["travelledTimeNormalized"]["std"] = delta_T_std_norm_first
	DATA["travelledTimeNormalized"]["var"] = delta_T_var_norm_first

	data_summary_folder = LOGSdir.replace("/LOGS","")
	with lzma.open(Path(data_summary_folder + "/data_summary.xz"), "wb") as file:
		pickle.dump(DATA, file)

if __name__ == '__main__':
    main()
    plt.show()
