import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from damgaard2022MR import LTMImplementation, PlanningImplementation, AppraisalsImplementation, ReflectiveAttentionMechanismImplementation

from pathlib import Path
from sys import path
import os
from os.path import dirname as dir
path.append(dir(Path(path[0].replace("multiRobotPlanning", ""))))
__package__ = "probmind"

from cognition.probMindAgent import ProbMindAgent

class UniCycleRobotPlanning(ProbMindAgent):

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

        # def per_param_lr(module_name, param_name):
        #     if "a_alpha_rot_" in param_name:
        #         return {"lr": configs["lr_a_alpha_rot"]}
        #     elif "a_beta_rot_" in param_name:
        #         return {"lr": configs["lr_a_beta_rot"]}
        #     elif "a_alpha_trans_" in param_name:
        #         return {"lr": configs["lr_a_alpha_trans"]}
        #     elif "a_beta_trans_" in param_name:
        #         return {"lr": configs["lr_a_beta_trans"]}
        #     else:
        #         return {"lr": configs["lr"]}
        # optimizer = pyro.optim.Adam(per_param_lr)

        self.model_error = configs["model_error"]
        self.a_support = configs["a_support"]
        self.alpha_init = configs["alpha_init"]
        self.beta_init = configs["beta_init"]

        self.T_delta = configs["T_delta"]
        self.t_delta = torch.tensor([configs["t_delta"]],dtype=torch.float)
        self.t_buffer = configs["t_buffer"]
        self.t_avg_planning = self.t_delta/4  # estimate of how long time is used to plan a new action
        self.t_count = 1

        appraisalsImplementationParams = {}
        appraisalsImplementation = AppraisalsImplementation(appraisalsImplementationParams)

        PlanningParams = {}
        PlanningParams["ID"] = robot_ID
        PlanningParams["alpha_init"] = configs["alpha_init"]
        PlanningParams["beta_init"] = configs["beta_init"]
        PlanningParams["svi_epochs"] = configs["svi_epochs"]
        PlanningParams["a_support"] = configs["a_support"]
        PlanningParams["model_error"] = configs["model_error"]
        PlanningParams["K"] = configs["K"]
        PlanningParams["lr_a_alpha_rot"] = configs["lr_a_alpha_rot"]
        PlanningParams["lr_a_beta_rot"] = configs["lr_a_beta_rot"]
        PlanningParams["lr_a_alpha_trans"] = configs["lr_a_alpha_trans"]
        PlanningParams["lr_a_beta_trans"] = configs["lr_a_beta_trans"]
        PlanningParams["lr"] = configs["lr"]
        PlanningParams["desirability_scale_factor"] = configs["desirability_scale_factor"]
        PlanningParams["radius"] = configs["r_radius"]
        self.planningImplementation = PlanningImplementation(PlanningParams, appraisalsImplementation)

        reflectiveAttentionMechanismImplementationParams = {}
        reflectiveAttentionMechanismImplementationParams["radius"] = configs["r_radius"]
        reflectiveAttentionMechanismImplementationParams["ID"] = robot_ID
        reflectiveAttentionMechanismImplementationParams["constraint_scale_factor"] = configs["constraint_scale_factor"]
        reflectiveAttentionMechanismImplementationParams["desirability_scale_factor"] = configs["desirability_scale_factor"]
        reflectiveAttentionMechanismImplementationParams["N_posterior_samples"] = configs["N_posterior_samples"]
        reflectiveAttentionMechanismImplementation = ReflectiveAttentionMechanismImplementation(appraisalsImplementation, self.planningImplementation, reflectiveAttentionMechanismImplementationParams)


        self.reset()

        super().__init__(reflectiveAttentionMechanismImplementation)


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


            dynamicParams = {}
            dynamicParams["msgs_received"] = msgs_received
            dynamicParams["z_goal"] = torch.tensor(goal_position, dtype=torch.float)
            dynamicParams["z_a_t_old"] = self.z_a_t_old
            dynamicParams["T"] = self.T
            dynamicParams["t_avg_planning"] = self.t_avg_planning
            dynamicParams["t_left_from_predicted_pose"] = self.t_left_from_predicted_pose
            dynamicParams["t_delta"] = self.t_delta

            ltm = LTMImplementation()
            z_a_tPlus_samples, z_s_tPlus_samples, k_samples, param_store = super().makePlan(self.T, self.T_delta, p_z_s_t, ltm, dynamicParams = dynamicParams)
            self.param_store = param_store


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
            act = self.planningImplementation.action_transforme(z_a_t).detach().cpu().numpy()

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
                print("Robot with ID " + str(self.ID) + " sends " + str(self.N_msgs_send) + " msgs pr. time step. Time pr. msgs: " + str(self.t_avg_planning) + " s. Msg received: " + str(len(msgs_received)))
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
                # a_alpha_trans = pyro.param("{}/a_alpha_trans_{}_{}".format(tau, self.ID, k)).detach()
                # a_alpha_rot = pyro.param("{}/a_alpha_rot_{}_{}".format(tau, self.ID, k)).detach()
                # a_beta_trans = pyro.param("{}/a_beta_trans_{}_{}".format(tau, self.ID, k)).detach()
                # a_beta_rot = pyro.param("{}/a_beta_rot_{}_{}".format(tau, self.ID, k)).detach()

                a_alpha_trans = self.param_store["{}/a_alpha_trans_{}_{}".format(tau, self.ID, k)].detach()
                a_alpha_rot = self.param_store["{}/a_alpha_rot_{}_{}".format(tau, self.ID, k)].detach()
                a_beta_trans = self.param_store["{}/a_beta_trans_{}_{}".format(tau, self.ID, k)].detach()
                a_beta_rot = self.param_store["{}/a_beta_rot_{}_{}".format(tau, self.ID, k)].detach()

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

    def p_z_s_t_single_robot(self, pose_mean, pose_L):
        meas_mean = torch.tensor(pose_mean, dtype=torch.float).detach()
        meas_L_ = torch.tensor(pose_L, dtype=torch.float).detach()
        _p_z_s_t = dist.MultivariateNormal(meas_mean, scale_tril=meas_L_)
        return _p_z_s_t


    #def _Planning__P_z_c_tau(self, tau, z_s_tau, z_s_tauMinus1):  
    #    # the __P_z_c_tau function in the planning module needs to be updated to work with dict states!
    #    # however this will require an update of the other examples!
    #    return torch.tensor([0.0])

