import numpy as np
from os import path
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import matplotlib.patches as patches
from matplotlib.legend_handler import HandlerPatch
import matplotlib
matplotlib.use('TkAgg') 
import warnings

import time

import multiprocessing

import gym
from gym import spaces
from gym.utils import seeding

from pathlib import Path
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

pi = 3.1415927410125732

colorWheel = {
    "red": "#EA4335",
    "orange": "#e09d2e",
    "yellow": "#b6d42a",
    "chartreuseGreen": "#57bd30",
    "green": "#34A853",
    "springGreen": "#2ec489",
    "cyan": "#2edbd2",
    "azure": "#37b6e9",
    "blue": "#4285F4",
    "violet": "#693ff2",
    "magenta": "#d43bef",
    "rose": "#ed3899",
}


colors = {
    "red": "#EA4335",
    "green": "#34A853",
    "blue": "#4285F4",
    "orange": "#e09d2e",
    "springGreen": "#2ec489",
    "violet": "#693ff2",
    "yellow": "#b6d42a",
    "cyan": "#2edbd2",
    "magenta": "#d43bef",
    "chartreuseGreen": "#57bd30",
    "azure": "#37b6e9",
    "rose": "#ed3899",
}


class multiRobotEnv(gym.Env):
    def __init__(self, N_Robots, config_path='configs/damgaard2022SVIFDPR/configs.yaml'):
        with open(Path(config_path), 'r') as stream:
            try:
                configs = yaml.load(stream, Loader=Loader)
            except yaml.YAMLError as exc:
                print(exc)

        self.configs = configs

        # self.action_space = self._get_action_space()
        # self.observation_space = self._get_observation_space()

        self.real_time_factor = configs["real_time_factor"]
        self.scenario = configs["scenario"]
        self.N_Robots = N_Robots
        self.plan_dist = configs["plan_dist"]
        self.goal_radius = configs["goal_radius"]

        self.r_radius = configs["r_radius"]
        self.r_diviation = configs["r_diviation_percentage"]/100

        self.receive_distance = configs["receive_distance"]
        self.plans = self.generate_plan()

        self.pos_samples = [None] * N_Robots

        self.render_receive_distance = configs["render_receive_distance"]

        # Consider revise/make dynamic?
        pos_std = configs["pos_error"]/3
        pos_var = pos_std*pos_std
        angle_std = np.deg2rad(configs["angle_error"])/3
        angle_var = angle_std*angle_std
        self.meas_cov = np.diag([pos_var,pos_var,angle_var])
        self.meas_L = np.linalg.cholesky(self.meas_cov)  # Cholesky decomposition
        self.estimation_scale_factor = configs["estimation_scale_factor"]

        self.is_sim_running = False

        self.reset()

    def get_sim_status(self):
        return self.is_sim_running

    def get_real_time_factor(self):
        return self.real_time_factor

    def are_there_any_collisions(self):
        for n in range(self.N_Robots-1):  # -1 since a robot cannot crash with itself
            if self.crash_flag[n]:
                return True

        return False

    def set_pos_samples(self, robot_ID, samples):
        self.pos_samples[robot_ID] = samples

    def simulator(self):
        tic = [time.time()] * self.N_Robots
        while True:
            try:  # when the env is reset some of the vars becomes None which causes errors.
                # propagate robots
                for robot_ID in range(self.N_Robots):
                    deltaT = (time.time() - tic[robot_ID])*self.real_time_factor
                    tic[robot_ID] = time.time()
                    if self.crash_flag[robot_ID]:
                        self.pose[robot_ID] = self.pose[robot_ID]  # no change if crashed
                    else:
                        self.pose[robot_ID] = self.unicycle_motion_model(self.pose[robot_ID], self.act[robot_ID], deltaT)

                # check for collisions
                for ID1 in range(self.N_Robots-1):
                    for ID2 in range(ID1+1, self.N_Robots):
                        dist_vector = np.array([self.pose[ID1][0] - self.pose[ID2][0],
                                                self.pose[ID1][1] - self.pose[ID2][1]])
                        dist = np.linalg.norm(dist_vector)

                        if dist <= self.r_robots[ID1]+self.r_robots[ID2]:
                            self.crash_flag[ID1] = True
                            self.crash_flag[ID2] = True

                time.sleep(0.005)
                self.sim_time = (time.time() - self.sim_time_start)*self.real_time_factor
            except:
                _ = True

    def send_msg(self, ID, msg):
        if self.is_sim_running:
            self.msgs[ID] = msg

    def generate_plan(self):
        plans = [None] * self.N_Robots
        if self.scenario == 0 or self.scenario == 1:
            angle_diff = pi/self.N_Robots  # antipodal switching

            for n in range(self.N_Robots):
                plan = []
                x1 = self.plan_dist*np.cos(angle_diff*n)
                y1 = self.plan_dist*np.sin(angle_diff*n)
                plan.append([x1,y1])

                x2 = self.plan_dist*np.cos(angle_diff*n+pi)
                y2 = self.plan_dist*np.sin(angle_diff*n+pi)
                plan.append([x2,y2])
                plans[n] = plan

        elif self.scenario == 2:
            angle_diff = pi/(self.N_Robots/2)  # circle-swapping antipodal switching

            for n in range(self.N_Robots):
                plan = []
                x1 = self.plan_dist*np.cos(angle_diff*n)
                y1 = self.plan_dist*np.sin(angle_diff*n)
                plan.append([x1,y1])

                x2 = self.plan_dist*np.cos(angle_diff*n+pi)
                y2 = self.plan_dist*np.sin(angle_diff*n+pi)
                plan.append([x2,y2])
                plans[n] = plan
        else:
            raise NameError('Unknown simulator scenario')

        return plans

    def reset_poses(self):
        for n in range(self.N_Robots):
            if self.scenario == 1:
                idx = np.random.randint(0, high=len(self.plans[n]))  # random initial pose
            if self.scenario == 0 or self.scenario == 2:
                if self.N_Robots > 2:
                    idx = n % 2  # evenly spaced initial pose
                else:
                    idx = 0

            if idx + 1 >= len(self.plans[n]):
                self.next_goal[n] = 0
            else:
                self.next_goal[n] = idx + 1

            x = self.plans[n][idx][0]
            y = self.plans[n][idx][1]
            heading = np.arctan2(self.plans[n][self.next_goal[n]][1]-y, self.plans[n][self.next_goal[n]][0]-x)  # heading towards the goal

            self.pose[n] = [x,  # x-coordinate
                            y,  # y-coordinate
                            heading]  # heading

            self.pose_est[n] = np.random.multivariate_normal([x,y,heading], self.meas_cov / self.estimation_scale_factor)    # simulate noise in the estimated pose

    def set_next_goal(self, robot_ID):
        if self.next_goal[robot_ID] is not None:  # else undefined behaviour
            if self.next_goal[robot_ID] + 1 >= len(self.plans[robot_ID]):
                self.next_goal[robot_ID] = 0
            else:
                self.next_goal[robot_ID] = self.next_goal[robot_ID] + 1

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def render_get(self):
        return self.goal_radius, self.plan_dist, self.N_Robots, self.plans, self.next_goal, self.pose, self.r_robots, self.pos_samples, self.receive_distance, self.render_receive_distance, self.sim_time, self.pose_est

    def time(self):
        return self.sim_time

    def get_poses(self):
        return self.pose

    def get_poses_est(self):
        return self.pose_est

    def get_current_goals(self):
        return self.next_goal

    def get_acts(self):
        return self.act

    def reset(self):
        self.is_sim_running = False
        print("######################## Resetting environment ########################")
        self.r_robots = np.random.uniform(low=self.r_radius*(1-self.r_diviation), high=self.r_radius*(1+self.r_diviation), size=self.N_Robots)
        self.pose = [None] * self.N_Robots
        self.pose_est = [None] * self.N_Robots
        self.act = [[0, 0]] * self.N_Robots
        self.next_goal = [None] * self.N_Robots
        self.msgs = [None] * self.N_Robots
        self.crash_flag = [False] * self.N_Robots
        self.N_msgs_received = [[] for _ in range(self.N_Robots)]
        self.reset_poses()
        self.pos_samples = [None] * self.N_Robots
        time.sleep(0.5)
        self.sim_time = 0
        self.sim_time_start = time.time()
        self.is_sim_running = True

    def receive_msgs(self, robot_ID):
        msgs_received = []
        
        for ID in range(self.N_Robots):
            if ID != robot_ID:
                dist_vector = np.array([self.pose[robot_ID][0] - self.pose[ID][0],
                                        self.pose[robot_ID][1] - self.pose[ID][1]])
                dist = np.linalg.norm(dist_vector)

                if dist <= self.receive_distance and self.msgs[ID] != None:  # only receive messages within a certain distance
                    msgs_received.append(self.msgs[ID])

        self.N_msgs_received[robot_ID].append(len(msgs_received))
        return msgs_received

    def get_N_msgs_received(self):
        return self.N_msgs_received

    def get_r_robots(self):
        return self.r_robots

    def step(self, action, robot_ID):
        try:
            self.act[robot_ID] = action

            if self.next_goal[robot_ID] is None:  #goal not set yet
                print("Goal not set...", flush=True)

            done = self._check_if_robot_has_reach_goal_zone(robot_ID)
            if done:
                self.set_next_goal(robot_ID)

            msgs = self.receive_msgs(robot_ID)

            if len(self.pose[robot_ID]) != 2:
                self.pose_est[robot_ID] = np.random.multivariate_normal(self.pose[robot_ID], self.meas_cov / self.estimation_scale_factor)    # simulate noise in the estimated pose
            else:
                self.pose_est[robot_ID] =  self.pose[robot_ID]

            if self.next_goal[robot_ID] is None:  #goal not set yet
                goal = self.pose_est[robot_ID]
            else:
                goal = self.plans[robot_ID][self.next_goal[robot_ID]]

            obs = [[self.pose_est[robot_ID], self.meas_L],
                    goal,
                    msgs]

            reward = 0.0
            info = {'is_success': done, 'is_crashed': self.crash_flag[robot_ID], 'sim_time': self.sim_time}
            return obs, reward, done, info
        except:  
            # an Exception sometime occur when resetting the env, however it is not really a problem...
            # it seems to have something to do with the "Goal not set..." printout
            warnings.warn("Exception encountered for robot with ID: " + str(robot_ID))
            obs = [[[0, 0, 0], self.meas_L],
                    [0, 0],
                    []]
            reward = 0.0
            info = {'is_success': False, 'is_crashed': False, 'sim_time': self.sim_time}
            return obs, reward, done, info

    def close(self):
        pass

    def _check_if_robot_has_reach_goal_zone(self, robot_ID):
        if self.next_goal[robot_ID] is None:
            print("'next_goal' of robot with ID" + str(robot_ID) + " is None")
            return False
        else:
            dist_vector = np.array([self.pose[robot_ID][0] - self.plans[robot_ID][self.next_goal[robot_ID]][0],
                                    self.pose[robot_ID][1] - self.plans[robot_ID][self.next_goal[robot_ID]][1]])
            goal_dist = np.linalg.norm(dist_vector)

            if goal_dist <= self.goal_radius:
                return True
            else:
                return False

    def _get_action_space(self):
        # action_space = spaces.Box(np.float32(np.array([-1, -1, -1])), np.float32(np.array([1, 1, 1])), dtype='float32')
        action_space = None
        return action_space

    def _get_observation_space(self): 
        #obs = self._get_obs()
        #observation_space = spaces.Box(np.float32(-np.inf), np.float32(np.inf), shape=obs.shape, dtype='float32')
        observation_space = None
        return observation_space

    def _get_obs(self):
        return 1 #obs

    def unicycle_motion_model(self, pos, act, deltaT):
        pos_new = np.zeros(3)

        pos_new[0] = pos[0] + np.cos(pos[2])*act[0]*deltaT
        pos_new[1] = pos[1] + np.sin(pos[2])*act[0]*deltaT
        pos_new[2] = pos[2] + act[1]*deltaT

        return pos_new



def render(env, mode='human', saveFolder=""):
    goal_radius, plan_dist, N_Robots, plans, next_goal, pose, r_robots, pos_samples, receive_distance, render_receive_distance, sim_time, pose_est = env.render_get()

    path = None
    msgReceivePatch = None
    if mode == "human":
        fig = plt.figure(100)
        #ax = plt.subplot()
        ax = plt.gca()
        ax.cla()

        xy_max = plan_dist + 2*goal_radius
        for ID in range(N_Robots):
            if abs(pose[ID][0])+2*r_robots[ID] > xy_max:
                xy_max = abs(pose[ID][0])+2*r_robots[ID]
            if abs(pose[ID][1])+2*r_robots[ID] > xy_max:
                xy_max = abs(pose[ID][1])+2*r_robots[ID]

        ax.set_xlim([-xy_max, xy_max])
        ax.set_ylim([-xy_max, xy_max])

        color_counter = 0
        for n in range(N_Robots):

            color = list(colors.values())[color_counter]

            if pos_samples[n] != None:  # draw planned trajectory samples
                z_s_tPlus_samples = pos_samples[n]
                for j in range(len(z_s_tPlus_samples)):
                    z_s_tPlus = []
                    for tau in range(len(z_s_tPlus_samples[j])):
                        pos_ = z_s_tPlus_samples[j][tau]
                        
                        # alpha = (len(z_s_tPlus_samples[j])-tau)/len(z_s_tPlus_samples[j]) * 0.1
                        # robot = plt.Circle((pos_[0], pos_[1]), robotRadius, color='black', alpha=alpha)
                        # plt.gca().add_patch(robot)

                        if tau == 0:
                            z_s_tPlus = pos_
                        else:
                            z_s_tPlus = np.vstack((z_s_tPlus, pos_))
                    path = ax.plot(z_s_tPlus[:, 0], z_s_tPlus[:, 1], color=color, zorder=6, label='Predicted Path Samples')


            for i in range(len(plans[n])):
                planZone = plt.Circle((plans[n][i][0],plans[n][i][1]), goal_radius, color=color, zorder=4, alpha=0.1, label='Next Goal')
                ax.add_patch(planZone)
            current_goalZone = plt.Circle((plans[n][next_goal[n]][0],plans[n][next_goal[n]][1]), goal_radius, color=color, zorder=5, label='Current Goal')
            ax.add_patch(current_goalZone)

            robotCirclePatch = plt.Circle((pose[n][0], pose[n][1]), r_robots[n], facecolor=color, edgecolor="black", zorder=10, label='robot')
            #robotCirclePatch = patches.Ellipse(xy=pose[n], width=r_robots[n]/2, height=r_robots[n]/2, facecolor=color, edgecolor="black", zorder=10, label='robot')
            ax.add_patch(robotCirclePatch)
            if render_receive_distance:
                msgReceivePatch = plt.Circle((pose[n][0], pose[n][1]), receive_distance, facecolor=(0,0,0,0), edgecolor=color, linestyle="--", zorder=10, label='Receive Distance')
                msgReceivePatch.set_alpha(None)
                ax.add_patch(msgReceivePatch)
            robotArrowPatch = patches.Arrow(pose[n][0],
                                            pose[n][1],
                                            (r_robots[n]*np.cos(pose[n][2])),
                                            (r_robots[n]*np.sin(pose[n][2])), 
                                            width=r_robots[n]/4,zorder=11,color="black")
            ax.add_patch(robotArrowPatch)
            scatter = plt.scatter(pose_est[n][0], pose_est[n][1], marker="+", color="black", zorder=15, label='Estimated Mean')

            ax.annotate(str(n), (pose[n][0], pose[n][1]+r_robots[n]*1.15), zorder=16)

            if color_counter == len(list(colors.values()))-1:
                color_counter = 0
            else:
                color_counter = color_counter + 1

        legend_elements = [planZone, 
                           current_goalZone, 
                           robotCirclePatch,
                           lines.Line2D([0], [0], color=color, lw=1, label='Predicted Path Samples'),
                           scatter]

        if msgReceivePatch != None:
            legend_elements.append(msgReceivePatch)
        ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.1), fancybox=True, shadow=False, ncol=len(legend_elements), handler_map={robotCirclePatch: HandlerRobot(), patches.Circle: HandlerCircle()})
        plt.title("Simulated Time {:.2f} s".format(sim_time))
        plt.draw()
        plt.pause(0.00000000001)


    else: # save data or create gif...
        print("error not implemented")

class HandlerRobot(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        d = 13
        r = d/2
        p1 = patches.Ellipse(xy=center, width=d, height=d, facecolor=colors["orange"], edgecolor="black")
        self.update_prop(p1, orig_handle, legend)
        p1.set_transform(trans)
        p3 = patches.Arrow(center[0],
                          center[1],
                          (r*np.cos(0)),
                          (r*np.sin(0)), 
                           width=r/4, color="black")
        #self.update_prop(p3, orig_handle, legend)
        #p3.set_transform(trans)
        return [p1,p3]

class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        d = 13 # ((height + ydescent) + (height + ydescent))/2
        p = patches.Ellipse(xy=center, width=d, height=d)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]


def test_agent(env, robot_ID, reset):
    act = [1/(robot_ID+1),0]
    while 1:
        obs, reward, done, info = env.step(act, robot_ID)
        if done:
            if act[0] > 0:
                act = [-1/(robot_ID+1),0]
            else:
                act = [1/(robot_ID+1),0]
        if reset.value:
           act = [1/(robot_ID+1),0]
        time.sleep(0.1)


def main():
    N_Robots = 12

    from multiprocessing.managers import BaseManager
    BaseManager.register('multiRobotEnv', multiRobotEnv)
    manager = BaseManager()
    manager.start()
    env = manager.multiRobotEnv(N_Robots)
    agent_reset = multiprocessing.Value("i", False)
    agent_reset.value = False

    p_agent = [None] * N_Robots
    for robot_ID in range(N_Robots):
        p_agent[robot_ID] = multiprocessing.Process(target=test_agent, args=(env, robot_ID, agent_reset))
        p_agent[robot_ID].daemon = True
        p_agent[robot_ID].start()

    p_sim = multiprocessing.Process(target=env.simulator, args=())
    p_sim.daemon = True
    p_sim.start()

    while True:
        render(env)
        time.sleep(0.01)

        if env.are_there_any_collisions():
            print("Collision detected. Resetting simulator!")
            agent_reset.value = True
            env.reset()
            time.sleep(0.1)
            agent_reset.value = False


if __name__ == '__main__':
    main()
