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


from pathlib import Path
from sys import path
import os
from os.path import dirname as dir
path.append(dir(Path(path[0].replace("robotPlanning", ""))))
__package__ = "probmind"

from cognition.planning import Planning

class RobotPlanning(Planning):

    def __init__(self, configs):
        K = configs["K"]  # K: number of options/trajectories to consider
        M = configs["M"]  # M: number of samples from each independent perception in calculation of the information gain
        N = configs["N"]  # N: number of LTM samples used in calculation of the information gain
        G = configs["G"]  # G: number of samples from each independent constraint
        L = configs["L"]  # L: number of past states to consider in progress

        self.N_posterior_samples = configs["N_posterior_samples"]

        desirability_scale_factor = configs["desirability_scale_factor"]
        progress_scale_factor = configs["progress_scale_factor"]
        info_gain_scale_factor = configs["info_gain_scale_factor"]
        svi_epochs = configs["svi_epochs"]

        # https://pyro.ai/examples/svi_part_iv.html
        initial_lr = configs["initial_lr"]
        gamma = configs["gamma"]  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / svi_epochs)
        optim_args = {'lr': initial_lr, 'lrd': lrd}
        optimizer = pyro.optim.ClippedAdam(optim_args)

        # Model specific params:
        standard_diviation = torch.tensor(configs["movement_3_sigma"] / 3)  # 99.7% of samples within a circle of 25 cm
        variance = standard_diviation * standard_diviation
        self.params["cov_s"] = variance * torch.eye(2)
        self.params["a_support"] = torch.tensor(configs["a_support"], dtype=torch.float)  # 1 m in each direction

        # lidar params
        lidarParams = {}
        lidarParams["meter2pixel"] = configs["meter2pixel"]
        lidarParams["z_max"] = torch.tensor([configs["lidar"]["range"]], dtype=torch.float)  # range in meters
        lidarParams["sigma_hit"] = torch.tensor(configs["lidar"]["sigma_hit"], dtype=torch.float)
        lidarParams["lambda_short"] = None  # not implemented
        lidarParams["N_lidar_beams"] = int(configs["lidar"]["fov"] / configs["lidar"]["resolution"])  # FOV is not used correctly. 360 degrees should always be used
        lidarParams["P_hit"] = configs["lidar"]["P_hit"]
        lidarParams["P_rand"] = configs["lidar"]["P_rand"]
        lidarParams["P_max"] = configs["lidar"]["P_max"]
        lidarParams["P_short"] = configs["lidar"]["P_short"]  # not implemented yet! => should always be zero
        self.params["lidarParams"] = lidarParams

        # constraint params
        self.params["P_z_C_scale"] = torch.tensor(configs["P_z_C_scale"], dtype=torch.float)
        self.params["d_min"] = torch.tensor([configs["robotRadius"] + configs["distance_buffer"]], dtype=torch.float)
        self.params["lidarParams_constraints"] = lidarParams.copy()

        self.constraint_subsampling  = configs["constraint_subsampling"]
        self.information_gain_subsampling = configs["information_gain_subsampling"]

        self.memorable_states = []

        super().__init__(K,
                         M,
                         N,
                         G,
                         L,
                         svi_epochs,
                         optimizer,
                         desirability_scale_factor=desirability_scale_factor,
                         progress_scale_factor=progress_scale_factor,
                         info_gain_scale_factor=info_gain_scale_factor,
                         consider_impasse=configs["consider_impasse"])

    def makePlan(self, tau, T_delta, p_z_s_t, map_grid_probabilities, return_mode="mean", p_z_g=None):
        self.map_grid_probabilities = map_grid_probabilities.detach()
        z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples = super().makePlan(tau, T_delta, p_z_s_t, N_posterior_samples=self.N_posterior_samples, p_z_g=p_z_g)

        if not (return_mode == "mean" or return_mode == "random"):
            return_mode == "random"

        if return_mode == "mean":
            # calculate the mean of the samples drawn from the posterior distribution:
            z_a_tauPlus_mean, z_s_tauPlus_mean, N_modes = self.calculate_state_action_means(z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, ForEachMode=True)

            if N_modes > 1:  # pick one mode randomly according to their assignment probability
                k = dist.Categorical(pyro.param('assignment_probs')).sample()
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
                #     if k == 0:
                #         if path_divergence > 0.5 and z_s2_infogain_prob > 0.8:
                #             self.memorable_states.append(poutine.trace(p_z_s_t).get_trace())
                #     else:
                #         if path_divergence > 0.5 and z_s1_infogain_prob > 0.8:
                #             self.memorable_states.append(poutine.trace(p_z_s_t).get_trace())

            else:
                z_a_tauPlus = z_a_tauPlus_mean[0]
                z_s_tauPlus = z_s_tauPlus_mean[0]

        elif return_mode == "random":  # return random sample
            z_a_tauPlus = z_a_tauPlus_samples[0]
            z_s_tauPlus = z_s_tauPlus_samples[0]

        # Transform actions
        for i in range(len(z_a_tauPlus)):
            z_a_tauPlus[i] = self.action_transforme(z_a_tauPlus[i]).detach()

        for j in range(self.N_posterior_samples):
            for i in range(len(z_a_tauPlus_samples[0])):
                z_a_tauPlus_samples[j][i] = self.action_transforme(z_a_tauPlus_samples[j][i]).detach()

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

    # ################### Abstract methods of the class Planning ####################
    def q_z_MB_tau(self, tau, z_s_tauMinus1, k):
        alpha_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        beta_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        a_alpha = pyro.param("a_alpha_{}".format(k), alpha_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
        a_beta = pyro.param("a_beta_{}".format(k), beta_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!

        _q_z_a_tau = dist.Beta(a_alpha, a_beta).to_event(1)
        z_a = pyro.sample("z_a", _q_z_a_tau)
        return z_a

    def p_z_MB_tau(self, tau, z_s_tau):
        # values should not be changed - use the params in "action_transforme" instead!
        a_min = torch.tensor([0., 0.], dtype=torch.float)
        a_max = torch.tensor([1., 1.], dtype=torch.float)
        _p_z_a_tau = dist.Uniform(a_min, a_max).to_event(1)
        z_a = pyro.sample("z_a", _p_z_a_tau)
        return z_a

    def p_z_s_tau(self, tau, z_s_tauMinus1, z_a_tauMinus1):
        mean = z_s_tauMinus1 + self.action_transforme(z_a_tauMinus1)
        cov = self.params["cov_s"]
        _p_z_s_tau = dist.MultivariateNormal(mean, cov)
        z_s = pyro.sample("z_s", _p_z_s_tau)
        return z_s

    def d_c_tau(self, tau, z_s_tau, z_LTM, z_PB_posterior):
        # z_s_tau: position
        # z_LTM: samples of distances to occupied map cells
        d_c_tau = []
        z_Map = z_LTM["z_Map"]
        z_lidar = z_PB_posterior["z_lidar"]

        # in this problem formulation both z_Map and z_lidar contain distance measures that
        # can be used directly as constraints.
        # using z_lidar we would include uncertainty about the lidar measurements, which
        # due to the lidar model includes uncertainty due to moving objects in the environment.
        # Thus if we want the robot to be extra carefull in its movement we should use the z_lidar,
        # but for static environment, we might as use z_Map.
        # distances = z_Map
        distances = z_lidar
        for key in distances:
            if distances[key] is None:  # if no cell is sampled to be occupied put distance to inf such that I_c=1
                d_c_tau.append(torch.tensor(float('inf')))
            else:
                d_c_tau.append(distances[key])

        return d_c_tau


    def p_z_LTM(self):
        map_grid_probabilities = self.map_grid_probabilities
        z_Map = p_z_Map_prior(map_grid_probabilities, self.params["lidarParams"])
        z_LTM = {}
        z_LTM["z_Map"] = z_Map
        return z_LTM

    def p_z_PB(self, tau, z_s_tau):
        p_z_Lidar_prior(self.params["lidarParams"])

    def p_z_PB_posterior(self, tau, z_s_tau, z_LTM):
        position = z_s_tau  # z_s_tau["position"]
        z_Map = z_LTM["z_Map"]
        z_lidar = p_z_Lidar_posterior(position, z_Map, self.params["lidarParams"])
        z_PB_posterior = {}
        z_PB_posterior["z_lidar"] = z_lidar
        return z_PB_posterior

    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau):
        position = z_s_tau  # z_s_tau["position"]
        PB_labels = lidar_generate_labels(self.params["lidarParams"], position, self.map_grid_probabilities, subsampling=self.information_gain_subsampling)
        return PB_labels

    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau):
        position = z_s_tau  # z_s_tau["position"]
        lidar_generate_labels(self.params["lidarParams"], position, self.map_grid_probabilities, subsampling=self.constraint_subsampling)
        # Consider incorporating funtionality to focus subsampling in the direction
        # of movement...

    def I_c_tilde(self, tau, d):
        # approximation to the indicator function used for distances
        # d: the distance to a constraint
        # _I_c_tilde: the approximation of the indicator function
        # here we have used an approximation shifted in d to account for the size (radius) of the robot
        # and to include an extra distance buffer
        _I_c_tilde = logistic(d, self.params["d_min"], self.params["P_z_C_scale"])
        return _I_c_tilde

    # ################### other methods ####################
    def action_transforme(self, z_a):
        # scaling parameters for the action
        a_offset = -self.params["a_support"] / torch.tensor([2.], dtype=torch.float)

        return self.params["a_support"] * z_a + a_offset