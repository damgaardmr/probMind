import numpy as np
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import pyro
import pyro.distributions as dist
from datetime import datetime
import os
import multiprocessing
import contextlib
import pickle
import lzma
import time
import psutil
import sys
from pathlib import Path

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from sys import path as pathSYS
from os.path import dirname as dir
pathSYS.append(dir(pathSYS[0].replace("robotPlanning", "misc"))) 

from robotPlanning import RobotPlanning

# from sim_evn import RobotExplorationT0 as SimEvn
from misc.HouseExpo.pseudoslam.envs.robot_exploration_ProbMind import RobotExplorationProbMind as SimEvn



def process_func(processID, dirName, DataDir, configs, env_config_file, Map_order, num_maps, t_max, reachGoalMode=False):
    with LoggingPrinter(Path(DataDir + "/terminal_log_" + str(processID) + ".txt")):
        print("Process with ID " + str(processID) + " started!")

        # set pyro's random seed for reproducability
        rand_seed = torch.randint(101, (1, 1))
        pyro.set_rng_seed(rand_seed[0][0])

        # instantiate the simulation environment
        with contextlib.redirect_stdout(open(os.devnull, 'w')):
            env_config_file_ = "../../../../../robotPlanning/" + env_config_file
            env = SimEvn(config_path=env_config_file_)
            env.reset(order=Map_order)

        meter2pixel = configs["meter2pixel"]
        robotRadius = configs["robotRadius"]  # env.sim.robotRadius / meter2pixel  # robot radius in meter
        lidar_range = configs["lidar"]["range"] # laser range in m
        plotWhileSim = configs["plotWhileSim"]

        # instantiate the planner
        agent = RobotPlanning(configs)

        # simulation params
        T_delta = configs["T_delta"]  # this was used for the full test

        act = np.zeros(3)  # start by doing nothing!

        dirName_ = dirName + "/" + env.sim.map_id
        n_map_sims = 0
        while os.path.exists(dirName_ + '/' + str(n_map_sims) + '.png'):
            n_map_sims = n_map_sims + 1


        t = 0
        data = {}
        data["explored_map"] = {}
        data["explored_map_collision"] = {}
        data["explored_map"]["t"] = []
        data["explored_map"]["map"] = []
        data["explored_map"]["planned_action_choosen"] = []
        data["explored_map"]["planned_state_choosen"] = []
        data["explored_map"]["planned_actions_samples"] = []
        data["explored_map"]["planned_states_samples"] = []
        data["explored_map"]["estimated_goal_mean"] = []
        data["explored_map"]["branchingStates_samples"] = []
        data["explored_map_collision"]["t"] = []
        data["explored_map_collision"]["map"] = []
        time_pr_iteration = []
        percentage_explored = []
        number_of_collisions = 0
        collisions = []
        maps_counter = 1
        z_s_tMinus = None
        z_a_tMinus = None
        reactiveAttentionMechanism = []
        while True:
            # print("################## RUNNING TIMESTEP " + str(t) + " ##################", flush=True)
            obs, reward, done, info = env.step(act)

            # make new plan
            tic = time.time()
            z_a_tPlus, z_s_tPlus_, z_a_tPlus_samples, z_s_tPlus_samples = agent.makePlan(t, T_delta, obs, configs, reachGoalMode=reachGoalMode)
            toc = time.time()
            time_pr_iteration.append(toc - tic)
            print("################## Process with ID " + str(processID) + " took " + str(toc - tic) + " s ##################")

            # convert plan to the format used by the simulator
            act[0] = z_a_tPlus[0].numpy()[0]
            act[1] = z_a_tPlus[0].numpy()[1]

            reactiveAttentionMechanism.append(agent.deliberateAttentionMechanism)

            goal_radius = configs["goal_zone_radius"]
            position = [obs[0][0], obs[0][1]]  # we only use the position not the heading
            map_grid_probabilities_np = obs[1]
            map_grid_probabilities = torch.from_numpy(map_grid_probabilities_np)
            map_grid_probabilities = torch.flip(map_grid_probabilities, [0])
            z_s_t = torch.tensor([position[0], position[1]], dtype=torch.float)
            if obs[2] is not None:
                goal_pos = torch.tensor([obs[2][0], obs[2][1]], dtype=torch.float)
            else:
                goal_pos = None

            if z_s_tMinus is None:
                z_s_tMinus = position
            else:
                z_s_tMinus = np.vstack((z_s_tMinus, position))
            
            if z_a_tMinus is None:
                z_a_tMinus = z_a_tPlus[0]
            else:
                z_a_tMinus = np.vstack((z_a_tMinus, z_a_tPlus[0]))

            branchingStates_samples = []
            for tau in agent.reflectiveAttentionMechanismImplementation.branchingStateIndexes:
                branchingStates_samples.append(agent.p_z_s_Minus_traces[tau].nodes["z_s"]["value"])


            # ####################### plotting #######################
            if plotWhileSim:
                mapShape = map_grid_probabilities.shape
                scalar = configs["imageScaling"]
                px = scalar * 1 / plt.rcParams['figure.dpi']  # pixel in inches
                fig = plt.figure(processID, figsize=(mapShape[1] * px, mapShape[0] * px))
                fig.clear()
                plt.title("Timestep t: " + str(t) + "   Percentage explored: " + "{:.2f}".format(env.sim.measure_ratio() * 100) + " % \n Map ID: " + env.sim.map_id + "         RAM: " + reactiveAttentionMechanism[-1])
                plt.imshow(map_grid_probabilities, cmap='binary', origin="upper", extent=[0, mapShape[1] / meter2pixel, 0, mapShape[0] / meter2pixel], aspect="auto", vmin=0.0, vmax=1.0)

                plt.gca().autoscale(enable=False, axis='both') # disable autoscaling when plottet elements are larger than the gridmap e.g. lidar range


                # draw past trajectory
                if t == 0:
                    plt.scatter(z_s_tMinus[0], z_s_tMinus[1], color="blue")
                else:
                    plt.plot(z_s_tMinus[:, 0], z_s_tMinus[:, 1], color="blue")

                if reachGoalMode:
                    # draw goal zone from goal_pos and goal_radius
                    #goalZone = plt.Circle((goal_pos[0].detach(), goal_pos[1].detach()), goal_radius, color='green')
                    goalZone_true = plt.Circle((env.goal_pos_in_m[0], env.goal_pos_in_m[1]), goal_radius, color='green')
                    plt.gca().add_patch(goalZone_true)

                    if obs[2] is not None:
                        goalZone_obs = plt.Circle((obs[2][0], obs[2][1]), 0.1, color='orange')
                        plt.gca().add_patch(goalZone_obs)

                # draw planned trajectory samples
                for j in range(len(z_s_tPlus_samples)):
                    z_s_tPlus = []
                    for tau in range(len(z_s_tPlus_samples[j])):
                        pos_ = z_s_tPlus_samples[j][tau].detach().cpu().numpy()
                        alpha = (len(z_s_tPlus_samples[j])-tau)/len(z_s_tPlus_samples[j]) * 0.1

                        robot = plt.Circle((pos_[0], pos_[1]), robotRadius, color='black', alpha=alpha)
                        plt.gca().add_patch(robot)

                        if tau == 0:
                            z_s_tPlus = pos_
                        else:
                            z_s_tPlus = np.vstack((z_s_tPlus, pos_))


                    plt.plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="black")

                # draw planned trajectory
                z_s_tPlus = []
                for tau in range(len(z_s_tPlus_)):
                    if tau == 0:
                        z_s_tPlus = z_s_tPlus_[tau].detach().cpu().numpy()
                    else:
                        z_s_tPlus = np.vstack((z_s_tPlus, z_s_tPlus_[tau].detach().cpu().numpy()))
                plt.plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color="orange")
                for i in range(len(z_s_tPlus)):
                    if i == 0:
                        lidar = plt.Circle((z_s_tPlus[i, 0], z_s_tPlus[i, 1]), lidar_range, fill=True, edgecolor=None, facecolor="blue", alpha=0.05, zorder=2, label="Lidar Range")
                    else:
                        lidar = plt.Circle((z_s_tPlus[i, 0], z_s_tPlus[i, 1]), lidar_range, fill=True, edgecolor=None, facecolor="blue", alpha=0.05, zorder=2)
                    plt.gca().add_patch(lidar)


                robot = plt.Circle((position[0], position[1]), robotRadius, color='blue')
                plt.gca().add_patch(robot)

                if info["is_crashed"] and t > 3:  # sometimes the simulator starts in a crashed state!!!
                    print("The agent crashed in map with ID: " + str(env.sim.map_id))
                    collisions.append(z_s_t)
                    # pos = z_s_t
                    # plt.scatter(pos[0].detach(), pos[1].detach(), color="red", s=100)
                    robot = plt.Circle((position[0], position[1]), robotRadius, color='red')
                    plt.gca().add_patch(robot)
                    number_of_collisions = number_of_collisions + 1
                    # save map
                    data["explored_map_collision"]["t"].append(t)
                    data["explored_map_collision"]["map"].append(map_grid_probabilities)
                    dirName_ = dirName + "/" + env.sim.map_id
                    if not os.path.exists(dirName_):
                        os.mkdir(dirName_)
                    fig.savefig(dirName_ + '/' + str(n_map_sims) + "_collision_" + str(number_of_collisions) + '.png')

                for col in collisions:
                    # plt.scatter(pos[0].detach(), pos[1].detach(), color="red", s=100)
                    robot = plt.Circle((col[0].detach(), col[1].detach()), robotRadius, color='red')
                    plt.gca().add_patch(robot)

                percentage_explored.append(env.sim.measure_ratio())

                for sample in branchingStates_samples:
                    branchingState = plt.Circle((sample[0], sample[1]), robotRadius, color='purple')
                    plt.gca().add_patch(branchingState)


                # allStateIndexes = agent.currentStateBranch.printTree()
                # for tau in allStateIndexes:
                #    stateTree_sample = agent.p_z_s_Minus_traces[tau].nodes["z_s"]["value"]
                #    plt.scatter(stateTree_sample[0], stateTree_sample[1], marker="*", color='black')

                # uncomment these lines the show plots
                # also comment out the line "matplotlib.use('Agg')"
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(.00001)




            data["explored_map"]["t"].append(t)
            data["explored_map"]["map"].append(map_grid_probabilities)
            data["explored_map"]["planned_action_choosen"].append(z_a_tPlus)  # [0])
            data["explored_map"]["planned_state_choosen"].append(z_s_tPlus_)  # [0])
            data["explored_map"]["planned_actions_samples"].append(z_a_tPlus_samples)
            data["explored_map"]["planned_states_samples"].append(z_s_tPlus_samples)
            data["explored_map"]["branchingStates_samples"].append(branchingStates_samples)
            if reachGoalMode:
                data["explored_map"]["estimated_goal_mean"].append(goal_pos)

            if t >= t_max or done:
                dirName_ = dirName + "/" + env.sim.map_id
                if not os.path.exists(dirName_):
                    os.mkdir(dirName_)

                dirName_ = dirName_ + "/process_" + str(processID)
                if not os.path.exists(dirName_):
                    os.mkdir(dirName_)

                if plotWhileSim:
                    fig.savefig(dirName_ + '/' + str(n_map_sims) + '.png')

                data["Map_ID"] = env.sim.map_id
                data["config_poseInit"] = env.sim.get_config_poseInit()
                data["lidar_range"] = lidar_range
                data["percentage_explored"] = percentage_explored
                data["trajectory"] = z_s_tMinus
                data["reactiveAttentionMechanism"] = reactiveAttentionMechanism
                data["actions"] = z_a_tMinus
                data["collisions"] = collisions  # not really necessary anymore.
                data["explored_map_collision"]["position"] = collisions
                data["meter2pixel"] = meter2pixel
                data["robotRadius"] = robotRadius
                data["trueMap"] = env.sim.world
                if reachGoalMode:
                    data["goal_pos"] = env.goal_pos_in_m
                    data["goal_radius"] = configs["goal_zone_radius"]
                metrics = {}
                metrics["true_map_area_in_pixels"] = np.sum(env.sim.world == env.sim.map_color['free'])
                metrics["time_pr_iteration"] = time_pr_iteration
                metrics["time_steps_used"] = t
                metrics["total_percentage_explored"] = env.sim.measure_ratio()
                metrics["number_of_collisions"] = number_of_collisions
                data["metrics"] = metrics

                # with open(dirName_ + '/' + str(n_map_sims) + ".p", "wb") as f:
                with lzma.open(Path(dirName_ + '/' + str(n_map_sims) + ".xz"), "wb") as f:
                    pickle.dump(data, f)

                # print("################## Process with ID: " + str(processID) + "  finished a map ##################")
                print("Time: " + str(time.time()))
                print("Map ID: " + env.sim.map_id)
                print("Percentage Explored: " + "{:.2f}".format(env.sim.measure_ratio() * 100) + "    Number of collisions: " + str(number_of_collisions) + "   Mean time pr. iteration: " + str(sum(time_pr_iteration) / len(time_pr_iteration)) + " True Map area: " + str(metrics["true_map_area_in_pixels"]))
                print("Finished " + str(maps_counter) + " out of " + str(num_maps) + " maps coresponding to: " + str(100 * maps_counter / num_maps) + " %")

                agent.reset()
                maps_counter += 1
                plt.close(processID)

                t = 0
                data = {}
                data["explored_map"] = {}
                data["explored_map_collision"] = {}
                data["explored_map"]["t"] = []
                data["explored_map"]["map"] = []
                data["explored_map"]["planned_action_choosen"] = []
                data["explored_map"]["planned_state_choosen"] = []
                data["explored_map"]["planned_actions_samples"] = []
                data["explored_map"]["planned_states_samples"] = []
                data["explored_map"]["estimated_goal_mean"] = []
                data["explored_map"]["branchingStates_samples"] = []
                data["explored_map_collision"]["t"] = []
                data["explored_map_collision"]["map"] = []
                time_pr_iteration = []
                percentage_explored = []
                number_of_collisions = 0
                collisions = []
                reactiveAttentionMechanism = []
                z_s_tMinus = None
                z_a_tMinus = None

                n_map_sims = 0
                while os.path.exists(dirName_ + '/' + str(n_map_sims) + '.png'):
                    n_map_sims = n_map_sims + 1

                if maps_counter > num_maps:
                    print("################## FINISHED ALL ASSIGNED MAPS ##################")
                    return

                with contextlib.redirect_stdout(open(os.devnull, 'w')):
                    env.reset(order=Map_order)
            else:
                t += 1


class LoggingPrinter:  # https://stackoverflow.com/questions/24204898/python-output-on-both-console-and-file
    def __init__(self, filename):
        self.out_file = open(filename, "w")
        self.old_stdout = sys.stdout
        # this object will take over `stdout`'s job
        sys.stdout = self

    # executed when the user does a `print`
    def write(self, text):
        self.old_stdout.write(text)
        self.old_stdout.flush()
        self.out_file.write(text)
        self.out_file.flush()
        os.fsync(self.out_file)

    # executed when `with` block begins
    def __enter__(self):
        return self

    # executed when `with` block ends
    def __exit__(self, type, value, traceback):
        # we don't want to log anymore. Restore the original stdout object.
        sys.stdout = self.old_stdout


def file_len(fname):
    i = 0
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main(args):
    os.system("taskset -p 0xffffffffff %d" % os.getpid())  # solves not fully utilization of cpu cores on ubuntu
    # remember to have enough f's relative to the number of cores!! https://stackoverflow.com/questions/31320194/python-multiprocessing-8-24-cores-loaded
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OMP error on mac M1
    # multiprocessing.freeze_support()  # <- may be required on windows

    # USAGE ...
    # python3 __main__.py -thread_start_ID 3 -total_cluster_threads 10 -cpu_cores 3
    cpu_cores = int(psutil.cpu_count(logical=False))  # use all cores by standard - only returns physical cores
    cpu_cores = 1
    thread_start_ID = 0
    total_cluster_threads = cpu_cores
    # config_folder = "configs/damgaard22Exploration"
    config_folder = "configs/damgaard22GoalSearch"
    #config_folder = "configs/damgaard22GoalSearch_experiment2/double_U_shape"
    #config_folder = "configs/damgaard22MultiModalActionPosterior"
    #config_folder = "configs/damgaard2022AKS"
    config_file = None
    print("args: " + str(args))
    # Treat args
    for i in range(len(args)):
        if args[i] == "-cpu_cores":  # the number of cpu cores that should be used - one process is started per core
            cpu_cores = int(args[i + 1])
        if args[i] == "-thread_start_ID":  # can be used to offset the process ID, i.e. for computer cluster use
            thread_start_ID = int(args[i + 1])
        if args[i] == "-total_cluster_threads":  # total number of threads that should/is running in the cluster - used to divide the map id
            total_cluster_threads = int(args[i + 1])
        if args[i] == "-config_file":  # the config file
            config_file = args[i + 1]
        if args[i] == "-config_folder":  # the config folder
            config_folder = args[i + 1]

    #config_file = "configs/damgaard22GoalSearch/configs.yaml"

    if config_file is None:
        config_file = config_folder + "/configs.yaml"


    folders_ = config_file.split("/")
    config_folder = folders_[0]
    for folder_ in folders_[1:len(folders_)-1]:
        config_folder = config_folder + "/" + folder_

    print(config_folder)

    with open(Path(config_file), 'r') as stream:
        try:
            configs = yaml.load(stream, Loader=Loader)
        except yaml.YAMLError as exc:
            print(exc)

    print("Using config file: " + config_file)

    Map_order = configs["Map_order"]
    t_max = configs["t_max"]
    reachGoalMode = configs["reachGoalMode"]


    N_processes = total_cluster_threads
    if thread_start_ID + cpu_cores >= total_cluster_threads:
        end_cluster_instances_id = total_cluster_threads
    else:
        end_cluster_instances_id = thread_start_ID + cpu_cores
    process_to_start = range(thread_start_ID, end_cluster_instances_id)

    now = datetime.now()
    date = now.strftime("%Y_%m_%d")
    current_time = now.strftime("%H_%M_%S")
    dirName = "DATA/date_" + date + "_time_" + current_time + "_thread_IDs_" + str(thread_start_ID) + "_" + str(end_cluster_instances_id - 1)

    if reachGoalMode:
        DataDir = dirName + "/sim_configs_goal"
    else:
        DataDir = dirName + "/sim_configs"

    # create folders to save data
    if not os.path.exists(dirName):
        os.mkdir(dirName)
    if not os.path.exists(DataDir):
        os.mkdir(DataDir)

    with LoggingPrinter(Path(DataDir + "/terminal_log_main" + ".txt")):
        print("Running version: 1.5.1")
        print("Total number of CPU's/Processes in cluster:  " + str(N_processes))
        print("Available CPU cores: " + str(cpu_cores))
        print("Starting " + str(len(process_to_start)) + " Processes")
        print("Starting processes with IDs: " + str(list(i for i in process_to_start)))
        print("Doing " + str(t_max) + " time steps pr. map.")

    map_id_set_file = config_folder + "/" + configs['map_id_set']

    num_maps = []
    with open(Path(map_id_set_file)) as infp:
        files = [open(Path(DataDir + "/map_ids_process_" + str(processID) + ".txt"), 'w+') for processID in range(N_processes)]
        for i, line in enumerate(infp):
            files[i % N_processes].write(line)
        for f in files:
            f.close()

    files = [Path(DataDir + "/map_ids_process_" + str(processID) + ".txt") for processID in range(N_processes)]
    for fname in files:
        num_maps.append(file_len(fname))

    jobs = []
    for processID in process_to_start:
        env_configs = {}
        env_configs['json_dir'] = configs['json_dir']
        env_configs['map_id_set'] = "../../../../" + config_folder + "/" + configs['map_id_set']
        env_configs['meter2pixel'] = configs['meter2pixel']
        env_configs['mode'] = configs['mode']
        env_configs["reachGoalMode"] = configs["reachGoalMode"]
        if configs["reachGoalMode"]:
            env_configs["goal_zone_radius"] = configs["goal_zone_radius"]
            env_configs["goal_zone_est_error_3_sigma"] = configs["goal_zone_est_error_3_sigma"]
            if "randomGoal" in configs:
                env_configs["randomGoal"] = configs["randomGoal"]
            if "goalPosition" in configs:
                env_configs["goalPosition"] = configs["goalPosition"]
            if "goal_always_observed" in configs:
                env_configs["goal_always_observed"] = configs["goal_always_observed"]
            else:
                env_configs["goal_always_observed"] = True
                configs["goal_always_observed"]  = True
        env_configs['continuesActions'] = True
        env_configs['obstacle'] = configs['obstacle']
        env_configs['robotRadius'] = configs['robotRadius']
        env_configs['stepLength'] = {}  # NOT RELEVENT! not used in this settings with "continuesActions: True"
        env_configs['stepLength']["linear"] = 1
        env_configs['stepLength']["angular"] = 1
        env_configs['startPose'] = configs['startPose']
        env_configs['resetRandomPose'] = configs['resetRandomPose']
        env_configs['laser'] = {}
        env_configs['laser']["range"] = configs['lidar']["range"]
        env_configs['laser']["fov"] = configs['lidar']["fov"]
        env_configs['laser']["resolution"] = configs['lidar']["resolution"]
        env_configs['laser']["noiseSigma"] = configs['lidar']["sigma_hit"]
        env_configs['slamError'] = configs['slamError']
        env_configs['stateSize'] = {}  # NOT RELEVENT - used to generate local map
        env_configs['stateSize']["x"] = 1  # NOT RELEVENT!
        env_configs['stateSize']["y"] = 1  # NOT RELEVENT!
        env_configs['map_id_set'] = "../../../../robotPlanning/" + DataDir + "/map_ids_process_" + str(processID) + ".txt"

        env_config_file = DataDir + "/sim_config_process_" + str(processID) + ".yaml"
        with open(Path(env_config_file), 'w') as stream:
            try:
                yaml.dump(env_configs, stream, default_flow_style=False, Dumper=Dumper)
            except yaml.YAMLError as exc:
                print(exc)

        p = multiprocessing.Process(target=process_func, args=(processID, dirName, DataDir, configs, env_config_file, Map_order, num_maps[processID], t_max, reachGoalMode))
        p.start()
        jobs.append(p)

    for job in jobs: # wait until all processes finishes
        job.join()


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args)