import pickle
import lzma
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import imageio
import tikzplotlib



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
	LOGSdir = "../damgaard2022SVIFDPR/LOGS"
	N_simulations = 50

	sim_time_all_sims = []
	dists_true_min_all_sims = []
	N_msgs_received_all_sims = []
	N_times_goal_reached_all_sims = []

	for sim_id in range(1, N_simulations+1):
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
					dists_true_.append(dist_true)
					
					dist_vector = np.array([poses_est[tau][n][0] - poses_est[tau][m][0],
											poses_est[tau][n][1] - poses_est[tau][m][1]])
					dist_est = np.linalg.norm(dist_vector) - r_robots[n] - r_robots[m]
					dists_est_.append(dist_est)
			dists_true.append(dists_true_)
			dists_true_min.append(np.min(dists_est_))
			dists_est.append(dists_true_)
			dists_est_min.append(np.min(dists_est_))

		dists_true_min_all_sims.append(dists_true_min)

		# ########################### Msg received calc ###########################
		N_msgs_received_all_robots = sum(N_msgs_received, [])
		N_msgs_received_all_sims.append(N_msgs_received_all_robots)

		# # ########################### goal reached calc ###########################
		current_goals_ = [[] for _ in range(N_robots)]
		for tau in range(len(current_goals)):
			for n in range(N_robots):
				current_goals_[n].append(current_goals[tau][n])

		N_times_goal_reached = []
		for n in range(N_robots):
			N_times_goal_reached.append(np.logical_xor(current_goals_[n][1:],current_goals_[n][:-1]).sum())

		N_times_goal_reached_all_sims.append(N_times_goal_reached)


	# # ########################### Seperating distance plot ###########################
	minimum_seperating_distance_pr_sim = []
	for sim_id in range(len(sim_time_all_sims)):
		minimum_seperating_distance_pr_sim.append(np.min(dists_true_min_all_sims[sim_id]))

	#print(minimum_seperating_distance_pr_sim)
	minimum_ = np.min(minimum_seperating_distance_pr_sim)
	mean = np.mean(minimum_seperating_distance_pr_sim)
	std = np.std(minimum_seperating_distance_pr_sim)
	var = np.var(minimum_seperating_distance_pr_sim)

	print("Minimum Seperating Distance pr sim mean: " + str(mean) + " +- " + str(std) + " std. (var: " + str(var) + ")")
	print("minimum of minimums ... : " + str(minimum_))
	fig, ax = plt.subplots(figsize=[6.4, 6.4/2])
	for sim_id in range(len(sim_time_all_sims)):
		if sim_id == 0:
			plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["blue"], label="MSD")
		else:
			plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["blue"])

	# plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["black"], linewidth=3, label="Seperation Distance")
	plt.plot([sim_time[0], sim_time[-1]], [0, 0], linewidth=3, color=colors["black"], label="Collision")
	plt.plot([sim_time[0], sim_time[-1]], [mean, mean], linewidth=2, linestyle="dotted", color=colors["black"], label="MSD Mean")
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
	plt.xlabel("Number of Times Reached [1]")
	plt.ylabel("Count [1]")

	tikzFolder = LOGSdir
	tikz_file_path = tikzFolder + "/" + "tikz_goals_reached_histogram.tikz"
	tikzplotlib.save(tikz_file_path)
	# modify generated tikz file
	width_str = "\\linewidth"
	plot_width = 1.0  # *width_str
	plot_height = 0.4  # *width_str
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


	plt.show()




if __name__ == '__main__':
    main()
