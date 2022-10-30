import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from lidarmodel import p_z_Lidar_prior, p_z_Lidar_posterior, p_z_Map_prior, lidar_generate_labels
from barrierfunctions import logistic


from damgaard2022AKS import LTMImplementation, PlanningImplementation, AppraisalsImplementation
from damgaard2022AKS import ReflectiveAttentionMechanismImplementation as ReflectiveAttentionMechanismImplementationAKS
from damgaard2022RGS import ReflectiveAttentionMechanismImplementation as ReflectiveAttentionMechanismImplementationRGS

from pathlib import Path
from sys import path
import os
from os.path import dirname as dir
path.append(dir(Path(path[0].replace("robotPlanning", ""))))
__package__ = "probmind"

from cognition.probMindAgent import ProbMindAgent


class RobotPlanning(ProbMindAgent):
    def __init__(self, configs):
        # lidar params
        lidarParams = {}
        lidarParams = configs["lidar"]
        lidarParams["meter2pixel"] = configs["meter2pixel"]
        lidarParams["z_max"] = torch.tensor([configs["lidar"]["range"]], dtype=torch.float)  # range in meters
        lidarParams["sigma_hit"] = torch.tensor(configs["lidar"]["sigma_hit"], dtype=torch.float)
        lidarParams["N_lidar_beams"] = int(configs["lidar"]["fov"] / configs["lidar"]["resolution"])  # FOV is not used correctly. 360 degrees should always be used
        self.params["lidarParams"] = lidarParams

        # constraint params
        self.params["lidarParams_constraints"] = lidarParams.copy()

        appraisalsImplementationParams = configs["AppraisalParams"]
        appraisalsImplementationParams["lidarParams"] = lidarParams
        appraisalsImplementationParams["desirability_scale_factor"] = torch.tensor([configs["AppraisalParams"]["desirability_scale_factor"]], dtype=torch.float)
        appraisalsImplementationParams["progress_scale_factor"] = torch.tensor([configs["AppraisalParams"]["progress_scale_factor"]], dtype=torch.float)
        appraisalsImplementationParams["info_gain_scale_factor"] = torch.tensor([configs["AppraisalParams"]["info_gain_scale_factor"]], dtype=torch.float)
        appraisalsImplementationParams["d_min"] = torch.tensor([configs["robotRadius"] + configs["AppraisalParams"]["distance_buffer"]], dtype=torch.float)
        appraisalsImplementationParams["information_gradient_multiplier"] = torch.tensor(configs["AppraisalParams"]["information_gradient_multiplier"])
        appraisalsImplementationParams["constraint_gradient_multiplier"] = torch.tensor(configs["AppraisalParams"]["constraint_gradient_multiplier"])
        appraisalsImplementationParams["P_z_C_scale"] = torch.tensor(configs["AppraisalParams"]["P_z_C_scale"], dtype=torch.float)

        appraisalsImplementation = AppraisalsImplementation(appraisalsImplementationParams)
        self.planningImplementation = PlanningImplementation(configs["PlanningParams"], appraisalsImplementation)

        reflectiveAttentionMechanismImplementationParams = configs["ReflectiveAttentionMechanism"]["params"]
        
        if configs["ReflectiveAttentionMechanism"]["name"] == "damgaard2022RGS":
            reflectiveAttentionMechanismImplementation = ReflectiveAttentionMechanismImplementationRGS(appraisalsImplementation, self.planningImplementation, reflectiveAttentionMechanismImplementationParams)
        else:
            reflectiveAttentionMechanismImplementation = ReflectiveAttentionMechanismImplementationAKS(appraisalsImplementation, self.planningImplementation, reflectiveAttentionMechanismImplementationParams)
            
        self.N_posterior_samples = configs["ReflectiveAttentionMechanism"]["params"]["N_posterior_samples"]

        super().__init__(reflectiveAttentionMechanismImplementation)

    def makePlan(self, tau, T_delta, obs, configs, reachGoalMode=False):
        return_mode = configs["return_Mode"]

        standard_diviation = torch.tensor(configs["initial_3_sigma"] / 3)  # 99.7% of samples within a circle of "initial_3_sigma" cm
        variance = standard_diviation * standard_diviation
        cov_s = variance * torch.eye(2)

        if reachGoalMode:
            goal_radius = configs["goal_zone_radius"]
            g_standard_diviation = torch.tensor(goal_radius / 3)  # 99.7% of samples within a circle with goal_radius
            g_variance = g_standard_diviation * g_standard_diviation
        else:
            p_z_g = None


        # convert obs from sim to the format used in the agent
        position = [obs[0][0], obs[0][1]]  # we only use the position not the heading
        map_grid_probabilities_np = obs[1]
        map_grid_probabilities = torch.from_numpy(map_grid_probabilities_np)
        map_grid_probabilities = torch.flip(map_grid_probabilities, [0])

        z_s_t = torch.tensor([position[0], position[1]], dtype=torch.float)

        def p_z_s_t():
            z_s_t_ = z_s_t.detach()
            cov_s_ = cov_s.detach()
            _p_z_s_t = dist.MultivariateNormal(z_s_t_, cov_s_)
            z_s = pyro.sample("z_s", _p_z_s_t)
            return z_s

        if reachGoalMode:
            if obs[2] is not None:
                goal_pos = torch.tensor([obs[2][0], obs[2][1]], dtype=torch.float)
                def p_z_g():
                    z_g_ = goal_pos.detach()
                    cov_g = g_variance * torch.eye(2)
                    _p_z_g = dist.MultivariateNormal(z_g_, cov_g)
                    z_g = pyro.sample("z_s", _p_z_g)
                    return z_g
            else:
                goal_pos = None
                p_z_g = None
        else:
            goal_pos = None
            p_z_g = None

        ltm = LTMImplementation(map_grid_probabilities.detach(), self.params["lidarParams"])
        z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, param_store = super().makePlan(tau, T_delta, p_z_s_t, ltm, p_z_g=p_z_g)

        if not (return_mode == "mean" or return_mode == "random"):
            return_mode == "random"

        if return_mode == "mean":
            # calculate the mean of the samples drawn from the posterior distribution:
            z_a_tauPlus_mean, z_s_tauPlus_mean, N_modes = self.calculate_state_action_means(z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, ForEachMode=True)

            if N_modes > 1:  # pick one mode randomly according to their assignment probability
                unique, counts = torch.unique(torch.tensor(k_samples), return_counts=True)
                probs = counts/counts.sum()
                k = dist.Categorical(probs).sample()
                z_a_tauPlus = z_a_tauPlus_mean[k]
                z_s_tauPlus = z_s_tauPlus_mean[k]

                # if N_modes == 2:
                #     # remember states where we could have obtained information by picking the other options
                #     # generalize to more than two modes!
                #     z_s1 = z_s_tauPlus_mean[0][-1].detach()
                #     z_s2 = z_s_tauPlus_mean[-1][-1].detach()
                #     path_divergence = torch.sum(((z_s1 - z_s2)**2))

                #     tmp = self.information_gain_subsampling
                #     self.information_gain_subsampling = 10
                #     z_s1_infogain_prob = super().P_z_i_tau(z_s1)
                #     z_s2_infogain_prob = super().P_z_i_tau(z_s2)
                #     self.information_gain_subsampling = tmp
                #     print(path_divergence)
                #     print(z_s1_infogain_prob)
                #     print(z_s2_infogain_prob)

            else:
                z_a_tauPlus = z_a_tauPlus_mean[0]
                z_s_tauPlus = z_s_tauPlus_mean[0]

        elif return_mode == "random":  # return random sample
            z_a_tauPlus = z_a_tauPlus_samples[0]
            z_s_tauPlus = z_s_tauPlus_samples[0]

        # Transform actions
        for i in range(len(z_a_tauPlus)):
            z_a_tauPlus[i] = self.planningImplementation.action_transforme(z_a_tauPlus[i]).detach()

        for j in range(self.N_posterior_samples):
            for i in range(len(z_a_tauPlus_samples[0])):
                z_a_tauPlus_samples[j][i] = self.planningImplementation.action_transforme(z_a_tauPlus_samples[j][i]).detach()

        return z_a_tauPlus, z_s_tauPlus, z_a_tauPlus_samples, z_s_tauPlus_samples

    def calculate_state_action_means(self, z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, ForEachMode=True):
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

    def calculate_state_action_mean(self, z_a_tauPlus_samples, z_s_tauPlus_samples):
        N_posterior_samples = len(z_a_tauPlus_samples)

        z_a_tauPlus_mean = []
        for tau in range(len(z_a_tauPlus_samples[0])):
            z_a_tau = torch.zeros_like(z_a_tauPlus_samples[0][tau])
            for i in range(N_posterior_samples):
                z_a_tau = z_a_tau + z_a_tauPlus_samples[i][tau]
            z_a_tauPlus_mean.append(z_a_tau / N_posterior_samples)

        z_s_tauPlus_mean = []
        for tau in range(len(z_s_tauPlus_samples[0])):
            z_s_tau = torch.zeros_like(z_s_tauPlus_samples[0][tau])
            for i in range(N_posterior_samples):
                z_s_tau = z_s_tau + z_s_tauPlus_samples[i][tau]
            z_s_tauPlus_mean.append(z_s_tau / N_posterior_samples)

        return z_a_tauPlus_mean, z_s_tauPlus_mean


