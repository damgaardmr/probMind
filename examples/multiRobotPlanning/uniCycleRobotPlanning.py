import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro.contrib.autoname import scope
from pyro import poutine
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from pathlib import Path
from sys import path
import os
from os.path import dirname as dir
path.append(dir(Path(path[0].replace("multiRobotPlanning", ""))))
__package__ = "probmind"

from cognition.planning import Planning

class UniCycleRobotPlanning(Planning):

    def __init__(self, robot_ID, configs):
        K = configs["K"]  # K: number of options/trajectories to consider
        M = 0  # M: number of samples from each independent perception in calculation of the information gain
        N = 0  # N: number of LTM samples used in calculation of the information gain
        G = 0  # G: number of samples from each independent constraint
        L = 0  # L: number of past states to consider in progress

        self.configs = configs
        self.ID = robot_ID
        self.N_posterior_samples = configs["N_posterior_samples"]

        desirability_scale_factor = configs["desirability_scale_factor"]
        self.constraint_scale_factor = configs["constraint_scale_factor"]
        self.radius = configs["r_radius"]
        
        svi_epochs = configs["svi_epochs"]
        #optim_args = {"lr":configs["lr"]}
        #optimizer = pyro.optim.Adam(optim_args)

        def per_param_lr(module_name, param_name):
            if "a_alpha_rot_" in param_name:
                return {"lr": configs["lr_a_alpha_rot"]}
            elif "a_beta_rot_" in param_name:
                return {"lr": configs["lr_a_beta_rot"]}
            elif "a_alpha_trans_" in param_name:
                return {"lr": configs["lr_a_alpha_trans"]}
            elif "a_beta_trans_" in param_name:
                return {"lr": configs["lr_a_beta_trans"]}
            else:
                return {"lr": configs["lr"]}
        optimizer = pyro.optim.Adam(per_param_lr)

        self.model_error = configs["model_error"]
        self.a_support = configs["a_support"]
        self.alpha_init = configs["alpha_init"]
        self.beta_init = configs["beta_init"]

        self.T_delta = configs["T_delta"]
        self.t_delta = torch.tensor([configs["t_delta"]],dtype=torch.float)
        self.t_buffer = configs["t_buffer"]
        self.t_avg_planning = self.t_delta/4  # estimate of how long time is used to plan a new action
        self.t_count = 1

        self.reset()

        super().__init__(K,
                         M,
                         N,
                         G,
                         L,
                         svi_epochs,
                         optimizer,
                         desirability_scale_factor=desirability_scale_factor,
                         progress_scale_factor=0.0,  # not used!
                         info_gain_scale_factor=0.0,  # not used!
                         consider_impasse=False)

    def reset(self):
        self.T = 0
        self.include_t_in_avg = False
        self.t_old = 0
        self.t_left_from_predicted_pose = 0

        self.z_a_t_old = torch.tensor([0., 0.], dtype=torch.float)
        self.act_old = [0., 0.]
        self.z_s_tPlus_samples_old = None
        self.msg_old = None
        self.T_old = 0
        self.N_msgs_send = 0

    def makePlan(self, pose_mean, pose_L, goal_position, current_time, msgs_received, Break=False, return_mode="mean"):

        # pyro.clear_param_store()

        def p_z_s_t():
            z_s = {}
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.ID), self.p_z_s_t_single_robot(pose_mean, pose_L))

            if msgs_received != None:
                with poutine.block():  # we only need the samples!
                    with torch.no_grad():
                        for ID in range(len(msgs_received)):
                            pose_mean_ = msgs_received[ID][0]
                            pose_L_ = msgs_received[ID][1]
                            z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.ID,ID), self.p_z_s_t_single_robot(pose_mean_, pose_L_)).detach()
            return z_s


        self.msgs_received = msgs_received
        self.z_goal = torch.tensor(goal_position, dtype=torch.float)

        self.T = int(np.floor(current_time / self.t_delta))
        self.t_left_to_next_timestep = (self.T + 1) * self.t_delta - current_time


        if self.t_left_to_next_timestep - self.t_buffer < self.t_avg_planning:  # not enough time to plan a new action!
            self.include_t_in_avg = False
            return self.act_old, self.msg_old, self.z_s_tPlus_samples_old
        else:
            if self.include_t_in_avg:
                t_since_last = current_time - self.t_old
                self.t_count = self.t_count + 1
                t_avg_planning_tmp = self.t_avg_planning + (t_since_last - self.t_avg_planning)/self.t_count  # if we used this value right away we risk that self.t_left_from_predicted_pose < 0
            self.t_left_from_predicted_pose = self.t_left_to_next_timestep - self.t_avg_planning 
            self.t_old = current_time

            try:
                z_a_tPlus_samples, z_s_tPlus_samples, k_samples = super().makePlan(self.T, self.T_delta, p_z_s_t, N_posterior_samples=self.N_posterior_samples)
            except:
                with open(self.configs["DataDir"] + "/errors_" + str(self.ID) + ".txt", "w") as file:
                    file.write("Error while trying to run SVI! any negative time? \n")
                    file.write("Current_time: " + str(current_time) + "\n")
                    file.write("T: " + str(self.T) + "\n")
                    file.write("t_left_to_next_timestep: " + str(self.t_left_to_next_timestep) + "\n")
                    file.write("t_avg_planning: " + str(self.t_avg_planning) + "Hello \n")
                    file.write("t_left_from_predicted_pose: " + str(self.t_left_from_predicted_pose) + "\n")
                    file.write("t_delta: " + str(self.t_delta) + "\n")

            if not (return_mode == "mean" or return_mode == "random"):
                return_mode == "random"

            k = 0
            if return_mode == "mean":
                # calculate the mean of the samples drawn from the posterior distribution:
                z_a_tPlus_mean, z_s_tPlus_mean, N_modes = self.calculate_state_action_means(z_a_tPlus_samples, z_s_tPlus_samples, k_samples, ForEachMode=True)

                if N_modes > 1:  # pick one mode randomly according to their assignment probability
                    k = dist.Categorical(pyro.param('assignment_probs')).sample()
                    z_a_t = z_a_tPlus_mean[k][1]
                else:
                    z_a_t = z_a_tPlus_mean[0][1]

            elif return_mode == "random":  # return random sample
                z_a_t = z_a_tPlus_samples[0][1]

            if Break:
                z_a_t = 0.5*torch.ones(2)  # do nothing!
                for j in range(len(z_s_tPlus_samples)):
                    for tau in range(len(z_s_tPlus_samples[j])):
                        z_s_tPlus_samples[j][tau] = p_z_s_t()  # we should be standing still

            # Transform actions
            act = self.action_transforme(z_a_t).detach().cpu().numpy()

            msg = self.pack_msg(pose_mean, pose_L, p_z_s_t, z_a_t, k, Break)

            if self.include_t_in_avg and t_avg_planning_tmp < self.t_delta and t_since_last > 0:  # the last part of this "if" is included to handle when the sim time is reset
                self.t_avg_planning = t_avg_planning_tmp
                # print("ID: " + str(self.ID) + " current_time: " + str(current_time) + "    T: " + str(self.T) + "   t_since_last: " + str(t_since_last) + "   t_left_to_next_timestep: " + str(self.t_left_to_next_timestep) + "   t_avg_planning: " + str(self.t_avg_planning) + "   t_left_from_predicted_pose: " + str(self.t_left_from_predicted_pose) + "   t_left_to_next_timestep: " + str(self.t_left_to_next_timestep))

            self.include_t_in_avg = True

            self.z_a_t_old = z_a_t
            self.act_old = act
            self.z_s_tPlus_samples_old = z_s_tPlus_samples
            self.msg_old = msg

            self.N_msgs_send = self.N_msgs_send + 1

            if self.T > self.T_old:
                print("Robot with ID " + str(self.ID) + " sends " + str(self.N_msgs_send) + " msgs pr. time step. Time pr. msgs: " + str(self.t_avg_planning) + " s. Msg received: " + str(len(self.msgs_received)))
                self.N_msgs_send = 0
            self.T_old = self.T

            return act, msg, z_s_tPlus_samples

    def pack_msg(self, pose_mean, pose_L, p_z_s_t, z_a_t, k, Break):
        msg = []
        msg.append(pose_mean)
        msg.append(pose_L)
        msg.append(z_a_t.detach().cpu().numpy())

        a_tPlus_alpha = []
        a_tPlus_beta = []

        for tau in range(self.T + 2, self.T + self.T_delta):
            if Break:
                a_alpha = torch.tensor([100000,100000], dtype=torch.float)
                a_beta = torch.tensor([100000,100000], dtype=torch.float)
            else:
                a_alpha_trans = pyro.param("{}/a_alpha_trans_{}_{}".format(tau, self.ID, k)).detach()
                a_alpha_rot = pyro.param("{}/a_alpha_rot_{}_{}".format(tau, self.ID, k)).detach()
                a_beta_trans = pyro.param("{}/a_beta_trans_{}_{}".format(tau, self.ID, k)).detach()
                a_beta_rot = pyro.param("{}/a_beta_rot_{}_{}".format(tau, self.ID, k)).detach()
                a_alpha = torch.stack((a_alpha_trans, a_alpha_rot), 0)
                a_beta = torch.stack((a_beta_trans, a_beta_rot), 0)

            a_tPlus_alpha.append(a_alpha)
            a_tPlus_beta.append(a_beta)

        msg.append(a_tPlus_alpha)
        msg.append(a_tPlus_beta)

        msg.append(self.t_avg_planning)
        msg.append(self.t_left_from_predicted_pose)
        msg.append(self.z_a_t_old.detach())
        msg.append(self.radius)

        return msg

    def calculate_state_action_means(self, z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, ForEachMode=True):
        # update to take the mean of all keys in the dicts!
        if ForEachMode:
            k_samples_ = torch.stack(k_samples)
            k_unique = k_samples_.unique()
            N_modes = len(k_unique)
            z_a_tauPlus_mean = []
            z_s_tauPlus_mean = []
            for k in range(N_modes):
                index_mask = k_samples_ == k_unique[k]
                index_mask = index_mask.flatten().tolist()
                z_a_tauPlus_samples_k = [item for keep, item in zip(index_mask, z_a_tauPlus_samples) if keep]
                z_s_tauPlus_samples_k = [item for keep, item in zip(index_mask, z_s_tauPlus_samples) if keep]
                z_a_tauPlus_mean_k, z_s_tauPlus_mean_k = self.calculate_state_action_mean(z_a_tauPlus_samples_k, z_s_tauPlus_samples_k)
                z_a_tauPlus_mean.append(z_a_tauPlus_mean_k)
                z_s_tauPlus_mean.append(z_s_tauPlus_mean_k)
        else:
            z_a_tauPlus_mean, z_s_tauPlus_mean = self.calculate_state_action_mean(z_a_tauPlus_samples, z_s_tauPlus_samples)
            N_modes = 1

        return z_a_tauPlus_mean, z_s_tauPlus_mean, N_modes

    def calculate_state_action_mean(self, z_a_tauPlus_samples, z_s_tauPlus_samples, dict_key_state = "own_pose", dict_key_action = "own_action"):
        # update to take the mean of all keys in the dicts!
        N_posterior_samples = len(z_a_tauPlus_samples)

        z_a_tauPlus_mean = []
        for tau in range(len(z_a_tauPlus_samples[0])):
            z_a_tau = torch.zeros_like(z_a_tauPlus_samples[0][tau][dict_key_action])
            for i in range(N_posterior_samples):
                z_a_tau = z_a_tau + z_a_tauPlus_samples[i][tau][dict_key_action]
            z_a_tauPlus_mean.append(z_a_tau / N_posterior_samples)

        z_s_tauPlus_mean = []
        for tau in range(len(z_s_tauPlus_samples[0])):
            z_s_tau = torch.zeros_like(z_s_tauPlus_samples[0][tau][dict_key_state])
            for i in range(N_posterior_samples):
                z_s_tau = z_s_tau + z_s_tauPlus_samples[i][tau][dict_key_state]
            z_s_tauPlus_mean.append(z_s_tau / N_posterior_samples)

        return z_a_tauPlus_mean, z_s_tauPlus_mean

    # ################### overwrite parent class methods ####################
    def _Planning__WM_planning_logic(self, tau, T, P_impasse, z_s_tau, p_z_s_tau_trace, p_z_s_Minus, P_z_C_tau, P_z_C_accum, P_z_d = None):
        # The planning logic in the paper related to this simulation is not quit the same as for the planning idiom. Therefore we overwrite it.
        # Here we observe the optimality variable as well as the constraint variable at each time step
        # However we could easily have used to logic of the planning idiom instead! 

        with scope(prefix=str(tau)):
            if self.msgs_received != None:
                for ID in range(len(self.msgs_received)):
                    min_dist = self.radius + self.msgs_received[ID][8] # + 0.1
                    pyro.sample("c_{}_{}_{}".format(self.ID,ID,tau), dist.Bernoulli(self.constraint(z_s_tau["own_pose"], z_s_tau[str(ID)], min_dist)), obs=torch.tensor([0.],dtype=torch.float)) # constraint!

            P_z_o = self.cost(tau, z_s_tau)
            pyro.sample("x_o_{}".format(self.ID), dist.Bernoulli(P_z_o), obs=torch.tensor([1.], dtype=torch.float))

        return None

    #@torch.jit.script
    def constraint(self, own_pose, others_pose, min_dist):
        dist = torch.dist(own_pose[torch.tensor([0, 1])], others_pose[torch.tensor([0, 1])], p=2)
        if dist<= min_dist:
            return torch.tensor([1.],dtype=torch.float)
        else:
            dist = dist - min_dist
            return torch.exp(-dist*self.constraint_scale_factor) # the higher constant the closer we allow the robots

    def cost(self, tau, z_s_tau):
        pose = z_s_tau["own_pose"]
        C = torch.dist(pose[[0,1]], self.z_goal.index_select(0, torch.tensor([0, 1])),p=2)
        # if C > 2.0:  # this allows us to tune the "desirability_scale_factor" and learning rates for goals a specific distance away from the robot
        #     goal_vector = self.z_goal.index_select(0, torch.tensor([0, 1])).detach() - pose[[0,1]].detach()
        #     goal_vector_normalized = torch.nn.functional.normalize(goal_vector, dim=0).detach()
        #     goal_tmp = pose[[0,1]].detach() + 2.0*goal_vector_normalized.detach()
        #     C = torch.dist(pose[[0,1]], goal_tmp,p=2)

        P_z_o = torch.exp(-self.params["desirability_scale_factor"] * C)
        return P_z_o

    def _Planning__P_z_c_tau(self, tau, z_s_tau, z_s_tauMinus1):  
        # the __P_z_c_tau function in the planning module needs to be updated to work with dict states!
        # however this will require an update of the other examples!
        return torch.tensor([0.0])

    # ################### Abstract methods of the class Planning ####################
    def q_z_MB_tau(self, tau, z_s_tauMinus1, k):
        z_a = {}

        if tau == self.T:  # this is fixed while we are planning
            z_a["own_action"] = self.z_a_t_old
        else:
            alpha_init = torch.tensor(self.alpha_init, dtype=torch.float) # small preference for going forward initially 
            beta_init = torch.tensor(self.beta_init, dtype=torch.float)
            # parameters are not automatically scoped!
            a_alpha_trans = pyro.param(str(tau)+"/"+"a_alpha_trans_{}_{}".format(self.ID, k), alpha_init[0], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_alpha_rot = pyro.param(str(tau)+"/"+"a_alpha_rot_{}_{}".format(self.ID, k), alpha_init[1], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_beta_trans = pyro.param(str(tau)+"/"+"a_beta_trans_{}_{}".format(self.ID, k), beta_init[0], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_beta_rot = pyro.param(str(tau)+"/"+"a_beta_rot_{}_{}".format(self.ID, k), beta_init[1], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!

            # a_beta_trans = torch.exp(a_beta_trans/1000/5)  # can potentially make the actions more aggressive!
            # a_alpha_trans = a_alpha_trans*a_alpha_trans*a_alpha_trans  # can potentially make the actions more aggressive!

            a_alpha = torch.stack((a_alpha_trans, a_alpha_rot), 0)
            a_beta = torch.stack((a_beta_trans, a_beta_rot), 0)

            _q_z_a_tau = dist.Beta(a_alpha, a_beta).to_event(1)
            z_a["own_action"] = pyro.sample("z_a_{}".format(self.ID), _q_z_a_tau)

        return z_a

    def p_z_MB_tau(self, tau, z_s_tau):
        z_a = {}
        if tau == self.T:  # this is fixed while we are planning
            z_a["own_action"] = self.z_a_t_old
        else:
            # values should not be changed - use the params in "action_transforme" instead!
            a_min = torch.tensor([0., 0.], dtype=torch.float)
            a_max = torch.tensor([1., 1.], dtype=torch.float)
            _p_z_a_tau = dist.Uniform(a_min, a_max).to_event(1)
            z_a["own_action"] = pyro.sample("z_a_{}".format(self.ID), _p_z_a_tau)

        with poutine.block():  # we only need the samples!
            with torch.no_grad():
                if self.msgs_received != None:
                    for ID in range(len(self.msgs_received)):
                        if tau == self.T:  # this is fixed while we are planning
                            z_a[str(ID)] = self.msgs_received[ID][7]
                        elif tau == self.T + 1:  # this is fixed while we are planning
                            z_a[str(ID)] = self.msgs_received[ID][2]
                        else:
                            if tau-(self.T+2) <= len(self.msgs_received[ID][3]):
                                a_alpha_ = self.msgs_received[ID][3][tau-(self.T+2)]
                                a_beta_ = self.msgs_received[ID][4][tau-(self.T+2)]
                                z_a[str(ID)] = pyro.sample("z_a_{}_{}".format(self.ID, ID), dist.Beta(a_alpha_, a_beta_).to_event(1)).detach()
        return z_a

    def p_z_s_tau(self, tau, z_s_tauMinus1, z_a_tauMinus1):
        z_s = {}
        if tau == self.T:  # only propagate time until the end of planning
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.ID), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], self.t_avg_planning))
        if tau == self.T + 1:  # propagate time left until next timestep!
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.ID), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], self.t_left_from_predicted_pose))
        else:  # propagate time a whole timestep
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.ID), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], self.t_delta))


        with poutine.block():  # we only need the samples!
            with torch.no_grad():
                if self.msgs_received != None:
                    for ID in range(len(self.msgs_received)):
                        if str(ID) in z_a_tauMinus1:  # if the function is called from the guide "z_a_tauMinus1" does not exist!
                            if tau == self.T:  # only propagate time until the end of planning
                                t_avg_planning = self.msgs_received[ID][5]
                                z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.ID,ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], self.t_avg_planning)).detach()
                            elif tau == self.T + 1:  # propagate time left until next timestep!
                                t_left_from_predicted_pose = self.msgs_received[ID][6]
                                z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.ID,ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], self.t_left_from_predicted_pose)).detach()
                            else:  # propagate time a whole timestep
                                if tau-(self.T+2) <= len(self.msgs_received[ID][3]):
                                    z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.ID,ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], self.t_delta)).detach()

        return z_s

    def d_c_tau(self, tau, z_s_tau, z_LTM, z_PB_posterior):
        return None

    def p_z_LTM(self):
        return None

    def p_z_PB(self, tau, z_s_tau):
        return None

    def p_z_PB_posterior(self, tau, z_s_tau, z_LTM):
        return None

    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau):
        return []

    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau):
        return []

    def I_c_tilde(self, tau, d):
        return None

    # ################### other methods ####################
    # @torch.jit.script
    def action_transforme(self, z_a_t):
        # scaling parameters for the action
        a_support = torch.tensor(self.a_support, dtype=torch.float) # [m/s,rad/s] turtlebot3 burger
        a_offset = -a_support/2
        return a_support*z_a_t + a_offset

    # @torch.jit.script
    def f(self, z_s_tauMinus1, z_a_t, t_delta):
        # Simple Kinematic Motion model for uni-cycle robot with linear/angular velocity (Twist) input
        # The model is dicretized by the Euler method
        a = self.action_transforme(z_a_t)

        mean = torch.empty(3)

        mean[0] = z_s_tauMinus1[0] + torch.cos(z_s_tauMinus1[2])*a[0]*t_delta[0]
        mean[1] = z_s_tauMinus1[1] + torch.sin(z_s_tauMinus1[2])*a[0]*t_delta[0]
        mean[2] = z_s_tauMinus1[2] + a[1]*t_delta

        return mean

    def F(self, z_s_tauMinus1, z_a_tauMinus1, t_delta):
        mean = self.f(z_s_tauMinus1, z_a_tauMinus1, t_delta)

        M = t_delta*torch.tensor(self.model_error,dtype=torch.float)
        return dist.Uniform(mean-M, mean+M).to_event(1)


    def p_z_s_t_single_robot(self, pose_mean, pose_L):
        meas_mean = torch.tensor(pose_mean, dtype=torch.float).detach()
        meas_L_ = torch.tensor(pose_L, dtype=torch.float).detach()
        _p_z_s_t = dist.MultivariateNormal(meas_mean, scale_tril=meas_L_)
        return _p_z_s_t
