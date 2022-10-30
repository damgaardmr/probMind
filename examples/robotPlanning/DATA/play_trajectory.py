import pickle
import lzma
import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import math
import imageio

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault)

# latex formatting strings:
# https://matplotlib.org/stable/tutorials/text/usetex.html
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
# remember that special charathers in strings in figures have to have prober escaping: i.e. use "\%" instead of "%"
plt.rcParams['text.usetex'] = True
#plt.rc('font', family='times')

#plt.rcParams.update({"text.usetex": True,
#                    "font.family": "sans-serif",
#                    "font.sans-serif": ["Helvetica"]})


colors = {
    "yellow": "#FBBC05",
    "green": "#34A853",
    "red": "#EA4335",
    "blue": "#4285F4",
    "purple": "#410093",
    "pink": "#FF33FF",
    "oliveGreen": "#AFB83B",
    "black": "black",
}

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def animationPlot(map_grid_probabilities, mapShape, meter2pixel, reachGoalMode, goal_pos, goal_radius, est_goal_center, robotRadius, position, collisions, z_s_tMinus, z_s_tPlus_samples, z_s_tPlus_, decision_mode_idx, decision_modes, lidar_range, pastBrancheState_samples):

    plt.imshow(map_grid_probabilities, cmap='binary', origin="upper", extent=[0, mapShape[1] / meter2pixel, 0, mapShape[0] / meter2pixel], aspect="auto", vmin=0.0, vmax=1.0, zorder=0)
    plt.gca().autoscale(enable=False, axis='both') # disable autoscaling when plottet elements are larger than the gridmap e.g. lidar range

    for i in range(len(pastBrancheState_samples)):
        if i == 0:
            pastBrancheState = plt.Circle((pastBrancheState_samples[i][0].detach(), pastBrancheState_samples[i][1].detach()), robotRadius, alpha=0.2, color=colors["purple"], label="pastBrancheState")
        else:
            pastBrancheState = plt.Circle((pastBrancheState_samples[i][0].detach(), pastBrancheState_samples[i][1].detach()), robotRadius, alpha=0.2, color=colors["purple"])
        plt.gca().add_patch(pastBrancheState)

    if reachGoalMode:
        # draw goal zone from goal_pos and goal_radius
        goalZone = plt.Circle((goal_pos[0], goal_pos[1]), goal_radius, color=colors["yellow"], label="True Goal Zone")
        plt.gca().add_patch(goalZone)
        if est_goal_center is not None:
            plt.scatter(est_goal_center[0], est_goal_center[1], marker="P", color="black", label="Est. Goal Mean", zorder=3)
        else:
            plt.scatter(0, 0, marker="P", color="black", label="Est. Goal Mean", zorder=3, alpha=0.0)  # only to get label

    for i in range(len(collisions)):
        if i == 0:
            robot = plt.Circle((collisions[i][0].detach(), collisions[i][1].detach()), robotRadius, color=colors["red"], label="Collision")
        else:
            robot = plt.Circle((collisions[i][0].detach(), collisions[i][1].detach()), robotRadius, color=colors["red"])
        plt.gca().add_patch(robot)


    # draw past trajectory
    if z_s_tMinus is None:
        z_s_tMinus = position
        plt.scatter(z_s_tMinus[0], z_s_tMinus[1], color=colors["green"])
    else:
        z_s_tMinus = np.vstack((z_s_tMinus, position))
        for tau in range(1,len(z_s_tMinus)-1):
            color = to_rgba(list(colors.values())[decision_mode_idx[tau]])
            plt.plot(z_s_tMinus[[tau-1,tau], 0], z_s_tMinus[[tau-1,tau], 1], color=color, linestyle='--')
        tau = len(z_s_tMinus)-1
        #plt.plot(z_s_tMinus[[tau-1,tau], 0], z_s_tMinus[[tau-1,tau], 1], color=colors["green"], linestyle='--', label=r"$Z_{\textrm{s}}^{\{0:t\}}$")

        for i in range(len(decision_modes)):
            color = to_rgba(list(colors.values())[i])
            if plt.rcParams['text.usetex']:
                label = r"$Z_{\textrm{s}}^{\{0:t\}}$ " + str(decision_modes[i]).replace("_"," ")
            else:
                label = str(decision_modes[i]).replace("_"," ")
            plt.plot(0.0, 0.0, color=color, linestyle='--', label=label)

    # draw planned trajectory samples
    for j in range(len(z_s_tPlus_samples)):
        z_s_tPlus = []
        for tau2 in range(len(z_s_tPlus_samples[j])):
            if tau2 == 0:
                z_s_tPlus = z_s_tPlus_samples[j][tau2]
            else:
                z_s_tPlus = np.vstack((z_s_tPlus, z_s_tPlus_samples[j][tau2].detach().cpu().numpy()))
        if j == len(z_s_tPlus_samples)-1:
            if plt.rcParams['text.usetex']:
                label = r"$Z_{\textrm{s}}^{\{t \}^{+} ,\{i_{a}\}}$"
            else:
                label = "None"
            plt.plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color=colors["green"], label=label)
        else:
            plt.plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color=colors["green"])

    robot = plt.Circle((position[0], position[1]), robotRadius, fill=True, edgecolor=colors["black"], facecolor=colors["green"], zorder=3)
    plt.gca().add_patch(robot)
    #z_s_t = z_s_tPlus_[0].detach().cpu().numpy()

    if plt.rcParams['text.usetex']:
        label = r"$E_{p\left(Z_{\textrm{s}}^{\{t\}}\right)}\left[Z_{\textrm{s}}^{\{t\}}\right]$"
    else:
        label = "None"
    plt.plot(position[0], position[1], ".", color=colors["black"], zorder=4, label=label)

    # draw planned trajectory
    start = 0
    if isinstance(z_s_tPlus_, list):
        z_s_tPlus = []
        for tau2 in range(start,len(z_s_tPlus_)):
            if tau2 == start:
                z_s_tPlus = z_s_tPlus_[tau2].detach().cpu().numpy()
            else:
                z_s_tPlus = np.vstack((z_s_tPlus, z_s_tPlus_[tau2].detach().cpu().numpy()))
        if plt.rcParams['text.usetex']:
            label=r"$Z_{\textrm{s}}^{\{t \}^{+},*}$"
        else:
            label = "None"
        plt.plot(z_s_tPlus[1:, 0], z_s_tPlus[1:, 1], "*", color=colors["black"], zorder=4, label=label)

        for i in range(len(z_s_tPlus)):
            if i == 0:
                if plt.rcParams['text.usetex']:
                    label = r"Lidar Range at $Z_{\textrm{s}}^{\{t \}^{+},*}$"
                else:
                    label = "None"
                lidar = plt.Circle((z_s_tPlus[i, 0], z_s_tPlus[i, 1]), lidar_range, fill=True, edgecolor=None, facecolor=colors["blue"], alpha=0.2, zorder=2, label=label)
            else:
                lidar = plt.Circle((z_s_tPlus[i, 0], z_s_tPlus[i, 1]), lidar_range, fill=True, edgecolor=None, facecolor=colors["blue"], alpha=0.2, zorder=2)
            plt.gca().add_patch(lidar)





def main():
    framerate = 100 #1 / 0.05
    scalar = 2  # scaling of plot
    n_images_to_same = 8
    SAVE_AS_BAD_EXAMPLE = False
    CREATE_GIF = True
    x_limits = None
    y_limits = None
    # x_limits = [6,16]  # limits on the plottet area
    # y_limits = [3,8.75]  # limits on the plottet area
    process_folder = None
    repetitionNumber = None

    DATAdir = "play_trajectory_example/Exploration"  # folder for which simulation data is saved
    mapID = "7fb9c9203cb8c4404f4af1781f1c6999"  # ID of the map for which the simulation should be replayed

    DATAdir = "play_trajectory_example/GoalSearch"  # folder for which simulation data is saved
    mapID = "3e5cc0e228c8a1bca9919a7c22c484d2"  # ID of the map for which the simulation should be replayed
    mapID = "b4e9112e72b9ba64b182841ae4ed443a"  # ID of the map for which the simulation should be replayed
    mapID = "9e1b0d8b332308d83101441d1b05f374"
    mapID = "30eb9263ce3efb9b5a943fc4161d4c6c"
    mapID = "3102ab88a439304a74fca0d26be703ef"
    mapID = "9497933d2a19318df74fd7197f515f1c"
    #mapID = "a730377aac9fddfe21ef643e31d11b88"

    #DATAdir = "play_trajectory_example/MultiModalActionPosterior"  # folder for which simulation data is saved
    #mapID = "7fb9c9203cb8c4404f4af1781f1c6999sim1"  # ID of the map for which the simulation should be replayed
    #mapID = "7fb9c9203cb8c4404f4af1781f1c6999sim2"  # ID of the map for which the simulation should be replayed
    #mapID = "7fb9c9203cb8c4404f4af1781f1c6999sim3"  # ID of the map for which the simulation should be replayed
    #mapID = "7fb9c9203cb8c4404f4af1781f1c6999"
    
    DATAdir = "date_2022_02_28_time_14_25_47_thread_IDs_0_0"  # folder for which simulation data is saved
    mapID = "empty"

    DATAdir = "date_2022_09_20_time_08_02_43_thread_IDs_0_2"
    mapID = "C_shape"
    mapID = "V_shape"
    #mapID = "double_U_shape"
    process_folder = "process_0"
    repetitionNumber = 1

    DATAdir = "date_2022_09_16_time_15_31_05_thread_IDs_0_2"
    mapID = "0c4526cfad6ac3d12c5ae5dc91327920"
    process_folder = "process_0"
    repetitionNumber = 1

    save_folder = DATAdir

    if process_folder is None:
        DATAdir = DATAdir + "/" + mapID
        save_folder = DATAdir + "/plots"
    else:
        DATAdir = DATAdir + "/" + mapID + "/" + process_folder
        save_folder = DATAdir + "/plots"
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    else:
        for f in os.listdir(save_folder):
            if f != "GIF":
                os.remove(os.path.join(save_folder, f))

    if CREATE_GIF:
        gif_folder = save_folder + "/GIF"
        if not os.path.exists(gif_folder):
            os.mkdir(gif_folder)
        else:
            for f in os.listdir(gif_folder):
                os.remove(os.path.join(gif_folder, f))


    DATAfiles = list_files(DATAdir)
    pickledFiles = []

    pickleFile = None
    for i in range(len(DATAfiles)):
        if mapID in DATAfiles[i] and DATAfiles[i].endswith(".xz"):
            pickleFile = DATAfiles[i]
            pickledFiles.append(pickleFile)
        # if DATAfiles[i].endswith(".xz") and DATAfiles[i].replace(".xz", ".p") not in DATAfiles:
        #    compressed_pickleFiles.append(DATAfiles[i])

    if repetitionNumber is not None:
        for pickleFile_ in pickledFiles:
            if pickleFile_.endswith("/"+str(repetitionNumber)+".xz"):
                pickleFile = pickleFile_
                break
    else:
        pickleFile = pickledFiles[0]

    print(pickleFile)

    if pickleFile is None:
        print("File with trajectory not found!")
        return
    else:
        f = lzma.open(pickleFile, 'rb')
        data = pickle.load(f)

        meter2pixel = data["meter2pixel"]
        percentage_explored = data["percentage_explored"]

        timesteps = len(data["explored_map"]["t"])
        print("timesteps: " + str(timesteps))

        delta_tau_to_save = (timesteps - 1) / (n_images_to_same - 1)
        tau_to_save = np.arange(0, timesteps + 1, delta_tau_to_save)
        n = 0

        # plotting
        mapShape = data["explored_map"]["map"][0].shape
        px = scalar * 1 / plt.rcParams['figure.dpi']  # pixel in inches
        #fig_animation = plt.figure(10, figsize=(mapShape[1] * px, mapShape[0] * px))
        fig_animation, axs_animation = plt.subplots(1,1, figsize=(mapShape[1] * px, mapShape[0] * px))

        z_s_tMinus = None

        #collisions = data["collisions"]

        if "goal_pos" in data:
            goal_pos = data["goal_pos"]
            goal_radius = data["goal_radius"]
            reachGoalMode = True
        else:
            goal_pos = None
            goal_radius = None
            reachGoalMode = False
        est_goal_center = None

        # robotRadius = 0.2
        robotRadius = data["robotRadius"]

        if "lidar_range" in data:
            lidar_range = data["lidar_range"]
        else:
            lidar_range = 4

        n_plots = len(tau_to_save)
        rows = 2
        plot_idx_in_row = 0
        row = 0
        plots_pr_row = math.ceil(n_plots/rows)
        fig, axs = plt.subplots(rows,plots_pr_row, figsize=[6.4, 6.4/2])

        #decision_modes = data["reactiveAttentionMechanism"]
        for tau in range(len(data["reactiveAttentionMechanism"])):
            if "_constraintAvoidance" in data["reactiveAttentionMechanism"][tau]:
                data["reactiveAttentionMechanism"][tau] = "constraintAvoidance"
            #data["reactiveAttentionMechanism"][tau] = data["reactiveAttentionMechanism"][tau].replace("_constraintAvoidance","")
        if "reactiveAttentionMechanism" in data:
            decision_modes = list(set(data["reactiveAttentionMechanism"]))
            print("decision_modes: " + str(decision_modes))


        distance_travelled_exploring = 0
        timesteps_used_exploring = 0

        decision_mode_idx = []
        for tau in range(timesteps):
            map_grid_probabilities = data["explored_map"]["map"][tau]
            position = data["trajectory"][tau]
            z_s_tPlus_samples = data["explored_map"]["planned_states_samples"][tau]
            z_s_tPlus_ = data["explored_map"]["planned_state_choosen"][tau]

            if "pastBrancheState_samples" in data["explored_map"]:
                pastBrancheState_samples = data["explored_map"]["pastBrancheState_samples"][tau]
            else:
                pastBrancheState_samples = []

            if "reactiveAttentionMechanism" in data:
                decision_mode = data["reactiveAttentionMechanism"][tau]
                decision_mode_idx.append(decision_modes.index(decision_mode))
                decision_mode = str(decision_mode).replace("_", " ")
                decision_mode_color = to_rgba(list(colors.values())[decision_mode_idx[tau]])

            if "Explore" in decision_mode:
                #print(decision_mode)
                p1 = data["trajectory"][tau-1]
                p2 = data["trajectory"][tau]
                dist = np.sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]))
                distance_travelled_exploring = distance_travelled_exploring + dist
                timesteps_used_exploring = timesteps_used_exploring + 1


            plt.sca(axs_animation)
            axs_animation.clear()
            plt.title("Timestep t: " + str(tau) + "\n Decision Mode: " + decision_mode + "      Map ID: " + mapID.replace("_", " "))

            # draw past trajectory
            if z_s_tMinus is None:
                z_s_tMinus = position
            else:
                z_s_tMinus = np.vstack((z_s_tMinus, position))


            collisions = []
            collisions_ = data["explored_map_collision"]
            for tau_ in collisions_["t"]:
                if tau >= tau_:
                    idx = collisions_["t"].index(tau_)
                    collisions.append(collisions_["position"][idx])


            if "goal_pos" in data:
                est_goal_center = data["explored_map"]["estimated_goal_mean"][tau]

            animationPlot(map_grid_probabilities, mapShape, meter2pixel, reachGoalMode, goal_pos, goal_radius, est_goal_center, robotRadius, position, collisions, z_s_tMinus, z_s_tPlus_samples, z_s_tPlus_, decision_mode_idx, decision_modes, lidar_range, pastBrancheState_samples)
            handles, labels = axs_animation.get_legend_handles_labels()
            if tau == 0:
                box = axs_animation.get_position()
                axs_animation.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
            axs_animation.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=int(np.ceil(len(labels)/2)))
            if x_limits != None and y_limits != None:
                axs_animation.set_xlim(x_limits)
                axs_animation.set_ylim(y_limits)


            if CREATE_GIF:
                gif_file_path = gif_folder + "/" + str(tau) + ".png"
                plt.savefig(gif_file_path, format='png', bbox_inches = "tight")

            # if tau >= tau_to_save[n]:
            #     print("saving fig number: " + str(n), flush=True)
            #     plt.savefig(save_folder + "/good_map_example_" + str(n) + '.pdf', format='pdf')

            #     plt.sca(axs[row,plot_idx_in_row])
            #     animationPlot(map_grid_probabilities, mapShape, meter2pixel, reachGoalMode, goal_pos, goal_radius, est_goal_center, robotRadius, position, collisions, z_s_tMinus, z_s_tPlus_samples, z_s_tPlus_, decision_mode_idx, decision_modes, lidar_range, pastBrancheState_samples)
            #     axs[row,plot_idx_in_row].set_title("{percentage:.2f}\%, t = {t}".format(percentage=percentage_explored[tau]*100,t=tau))
            #     if tau == tau_to_save[-1]:
            #         handles, labels = axs[row,plot_idx_in_row].get_legend_handles_labels()
            #         #fig.legend(handles, labels, loc='lower center', ncol=len(labels))
            #         #plt.subplots_adjust(bottom=0.2)
            #         axs[row,plot_idx_in_row].legend(loc="lower right", ncol=len(labels))

            #     n = n + 1

            #     if plot_idx_in_row == plots_pr_row-1:
            #         row = row + 1
            #         plot_idx_in_row = 0
            #     else:
            #         plot_idx_in_row = plot_idx_in_row + 1

            fig_animation.canvas.draw()
            fig_animation.canvas.flush_events()
            plt.pause(1/framerate)

        if CREATE_GIF: # Build GIF
            with imageio.get_writer( save_folder + "/" + str(mapID) + '.gif', mode='I', fps=5) as writer:
                for tau2 in range(len(os.listdir(gif_folder))):
                    image = imageio.imread(gif_folder + "/" + str(tau2) + ".png")
                    writer.append_data(image)
            for f in os.listdir(gif_folder):
                os.remove(os.path.join(gif_folder, f))
            os.rmdir(gif_folder)
            print("Gif created!")

        print("Total distance_travelled_exploring: " + str(distance_travelled_exploring))
        print("timesteps_used_exploring: " + str(timesteps_used_exploring))
        print("Dist pr. timestep exploring: " + str(distance_travelled_exploring/timesteps_used_exploring))

        plt.show()

if __name__ == '__main__':
    main()
