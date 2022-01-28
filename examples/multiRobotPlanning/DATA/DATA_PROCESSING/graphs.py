import pickle
import lzma
import os
import matplotlib.pyplot as plt
import numpy as np
import math
import imageio

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
	LOGSdir = "../date_2022_01_27_time_13_06_03/LOGS"
	N_simulations = 45

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
	mean = np.mean(minimum_seperating_distance_pr_sim)
	std = np.std(minimum_seperating_distance_pr_sim)
	var = np.var(minimum_seperating_distance_pr_sim)

	print("Minimum Seperating distance pr sim mean: " + str(mean) + " +- " + str(std) + " std. (var: " + str(var) + ")")
	plt.figure(1)
	for sim_id in range(len(sim_time_all_sims)):
		plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id]) #, color=colors["blue"])
	
	# plt.plot(sim_time_all_sims[sim_id], dists_true_min_all_sims[sim_id], color=colors["black"], linewidth=3, label="Seperation Distance")
	plt.plot([sim_time[0], sim_time[-1]], [0, 0], linewidth=3, linestyle="--", color=colors["black"], label="Collision")
	plt.legend()


	# # ########################### Msg received histogram ###########################
	N_msgs_received_all_sims = sum(N_msgs_received_all_sims, [])
	bins = np.arange(0, np.max(N_msgs_received_all_sims) + 1.5) - 0.5
	fig, ax = plt.subplots()
	_ = ax.hist(N_msgs_received_all_sims, bins)
	ax.set_xticks(bins + 0.5)

	# ########################### goal reached histogram ###########################
	N_times_goal_reached_all_sims = sum(N_times_goal_reached_all_sims, [])
	plt.figure(3)
	bins = np.arange(0, np.max(N_times_goal_reached_all_sims) + 1.5) - 0.5
	fig, ax = plt.subplots()
	_ = ax.hist(N_times_goal_reached_all_sims, bins)
	ax.set_xticks(bins + 0.5)


	plt.show()




if __name__ == '__main__':
    main()
