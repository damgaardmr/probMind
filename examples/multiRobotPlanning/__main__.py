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
from multiprocessing.managers import BaseManager
import contextlib
import pickle
import lzma
import time
import sys
from pathlib import Path

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from uniCycleRobotPlanning import UniCycleRobotPlanning
import env_multi_robot


def agent_process(env, robot_ID, reset, DataDir, configs):
    with LoggingPrinter(Path(DataDir + "/terminal_log_" + str(robot_ID) + ".txt")):
        # set pyro's random seed for reproducability
        rand_seed = torch.randint(101, (1, 1))
        pyro.set_rng_seed(rand_seed[0][0])

        agent = UniCycleRobotPlanning(robot_ID, configs)

        act = np.zeros(2)  # start by doing nothing!

        data = {}
        time_pr_iteration_true = []

        while True:
            obs, reward, done, info = env.step(act, robot_ID)

            # convert obs from sim to the format used in the agent
            pose_mean = obs[0][0]  # we only use the position not the heading
            pose_L = obs[0][1]  # we only use the position not the heading
            goal_position = obs[1]
            msgs_received = obs[2]
            time_sim = info["sim_time"]

            # make new plan
            tic = time.time()
            act, msg, z_s_tPlus_samples = agent.makePlan(pose_mean, pose_L, goal_position, time_sim, msgs_received)
            toc = time.time()
            time_pr_iteration_true.append(toc - tic)

            # if done:
            if reset.value:
                print("######################## Resetting agent " + str(robot_ID) + " ########################")
                agent.reset()
                act = np.zeros(2)  # start by doing nothing!
                time.sleep(1.0)
            else:
                env.send_msg(robot_ID, msg)

                # only for plotting...
                # convert samples to the format used by the simulator
                if z_s_tPlus_samples != None:
                    z_s_tPlus_samples_np = []
                    for j in range(len(z_s_tPlus_samples)):
                        z_s_tPlus_ = []
                        for tau in range(len(z_s_tPlus_samples[j])):
                            pose_ = z_s_tPlus_samples[j][tau]["own_pose"].detach().cpu().numpy()
                            z_s_tPlus_.append(pose_)

                        z_s_tPlus_samples_np.append(z_s_tPlus_)

                    env.set_pos_samples(robot_ID, z_s_tPlus_samples_np)


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
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def main():
    os.system("taskset -p 0xffffffffff %d" % os.getpid())  # solves not fully utilization of cpu cores on ubuntu
    # remember to have enough f's relative to the number of cores!! https://stackoverflow.com/questions/31320194/python-multiprocessing-8-24-cores-loaded
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # OMP error on mac M1
    # multiprocessing.freeze_support()  # <- may be required on windows

    # USAGE ...
    # python3 __main__.py -config_file "configs/damgaard2022SVIFDPR/configs.yaml"

    config_folder = "configs/damgaard2022SVIFDPR"
    config_file = config_folder + "/configs.yaml"
    
    # Treat args
    args = sys.argv[1:]
    for i in range(len(args)):
        if args[i] == "-config_file":  # the config file
            config_file = args[i + 1]

    now = datetime.now()
    date = now.strftime("%Y_%m_%d")
    current_time = now.strftime("%H_%M_%S")
    DataDir = "DATA/date_" + date + "_time_" + current_time

    if not os.path.exists(DataDir):
        os.mkdir(DataDir)

    with open(Path(config_file), 'r') as stream:
        try:
            configs = yaml.load(stream, Loader=Loader)
            configs["DataDir"] = DataDir
        except yaml.YAMLError as exc:
            print(exc)


    BaseManager.register('multiRobotEnv', env_multi_robot.multiRobotEnv)
    manager = BaseManager()
    manager.start()
    env = manager.multiRobotEnv(configs["N_Robots"])
    agent_reset = multiprocessing.Value("i", False)
    agent_reset.value = False

    p_agent = [None] * configs["N_Robots"]
    for robot_ID in range(configs["N_Robots"]):
        p_agent[robot_ID] = multiprocessing.Process(target=agent_process, args=(env, robot_ID, agent_reset, DataDir, configs))
        p_agent[robot_ID].daemon = True
        p_agent[robot_ID].start()

    p_sim = multiprocessing.Process(target=env.simulator, args=())
    p_sim.daemon = True
    p_sim.start()

    frame_counter = 0
    frametime = 1/configs["framerate"]*(1/env.get_real_time_factor())
    if not os.path.exists(DataDir + "/MEDIA"):
        os.mkdir(DataDir + "/MEDIA")

    log_folder = DataDir + "/LOGS"
    if not os.path.exists(log_folder):
        os.mkdir(log_folder)
    log_dict = {}
    log_dict["sim_time"] = []
    log_dict["robot_poses"] = []
    log_dict["robot_poses_est"] = []
    log_dict["current_goals"] = []

    simNumber = 1
    media_DATA_folder_path = DataDir + "/MEDIA/pngs_sim_" + str(simNumber)
    if not os.path.exists(media_DATA_folder_path):
        os.mkdir(media_DATA_folder_path)

    with LoggingPrinter(Path(DataDir + "/main_log.txt")):
        print("Starting simulations and rendering")
        
        while simNumber <= configs["N_simulations"]:

            log_dict["sim_time"].append(env.time())
            log_dict["robot_poses"].append(env.get_poses())
            log_dict["robot_poses_est"].append(env.get_poses_est())
            log_dict["current_goals"].append(env.get_current_goals())

            env_multi_robot.render(env)

            if configs["save_pngs_for_gif"] and  env.time() > frametime*frame_counter:
                png_DATA_file_path = media_DATA_folder_path + "/" + str(frame_counter) + ".png"
                plt.savefig(png_DATA_file_path, format='png', bbox_inches = "tight")
                frame_counter = frame_counter + 1

            if env.are_there_any_collisions() or env.time() >= configs["simulated_time_pr_sim"]:
                if env.are_there_any_collisions():
                    print("Collision detected in simulation " + str(simNumber) + " at simTime " + str(env.time()) + " s. Resetting simulator!")
                if env.time() >= configs["simulated_time_pr_sim"]:
                    print("Finished simulation " + str(simNumber) + ".")
                
                log_dict["r_robots"] = env.get_r_robots()
                log_dict["N_msgs_received"] = env.get_N_msgs_received()
                with lzma.open(Path(log_folder + '/log_sim_' + str(simNumber) + ".xz"), "wb") as f:
                    pickle.dump(log_dict, f)
                log_dict = {}
                log_dict["sim_time"] = []
                log_dict["robot_poses"] = []
                log_dict["robot_poses_est"] = []
                log_dict["current_goals"] = []

                simNumber = simNumber + 1
                if simNumber <= configs["N_simulations"]:
                    agent_reset.value = True
                    time.sleep(0.5)
                    env.reset()
                    time.sleep(0.5)
                    agent_reset.value = False
                    frame_counter = 0
                    media_DATA_folder_path = DataDir + "/MEDIA/pngs_sim_" + str(simNumber)
                    if not os.path.exists(media_DATA_folder_path):
                        os.mkdir(media_DATA_folder_path)
            
            time.sleep(0.02)

        print("Finished all simulations")
    
    for robot_ID in range(configs["N_Robots"]):
        p_agent[robot_ID].terminate()
    p_sim.terminate()

if __name__ == '__main__':
    main()