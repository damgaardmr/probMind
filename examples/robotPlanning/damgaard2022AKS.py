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
from cognition.ltm import LTM
from cognition.appraisals import Appraisals
from cognition.reflectiveAttentionMechanism import ReflectiveAttentionMechanism
#from cognition.deliberateAttentionMechanism import Damgaard2022AKS
from cognition.deliberateAttentionMechanism import DeliberateAttentionMechanism
from cognition.misc import probabilistic_OR_independent, probabilistic_AND_independent



class Damgaard2022AKS(DeliberateAttentionMechanism):
    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["progress"] = self.appraisals_.P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams)        
        P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams, mode="or")

        P_z_C_accum = torch.tensor([0.0])

        return P_alpha_tau, P_z_C_accum

    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        P_z_A1 = probabilistic_OR_independent([P_alpha_tau["information gain"], P_alpha_tau["progress"]])  # <-- the order of args might matter!
        P_z_A = probabilistic_AND_independent([P_z_A1, P_alpha_tau["constraints"]])

        return P_z_A


class LTMImplementation(LTM):
    def __init__(self, map_grid_probabilities, lidarParams):
        super().__init__()
        self.content["map_grid_probabilities"] = map_grid_probabilities
        self.lidarParams = lidarParams

    def p_z_LTM(self, dynamicParams):
        z_Map = p_z_Map_prior(self.content["map_grid_probabilities"], self.lidarParams)
        z_LTM = {}
        z_LTM["z_Map"] = z_Map
        return z_LTM


class PlanningImplementation(Planning):
    def __init__(self, 
                 params,
                 appraisalsImplementation):
        self.params = params

        standard_diviation = torch.tensor(params["movement_3_sigma"] / 3)  # 99.7% of samples within a circle of 25 cm
        variance = standard_diviation * standard_diviation
        self.params["cov_s"] = variance * torch.eye(2)
        self.params["a_support"] = torch.tensor(params["a_support"], dtype=torch.float)  # 1 m in each direction

        self.appraisalsImplementation = appraisalsImplementation

        super().__init__()

    def action_transforme(self, z_a):
        # scaling parameters for the action
        a_offset = -self.params["a_support"] / torch.tensor([2.], dtype=torch.float)

        return self.params["a_support"] * z_a + a_offset


    # ################### Abstract methods of the class Planning ####################
    def q_z_MB_tau(self, tau, z_s_tauMinus1, k, dynamicParams):
        #alpha_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        #beta_init = torch.tensor([[10000., 10000.], [10000., 10000.]], dtype=torch.float)
        #alpha_init = torch.tensor([[100., 100.], [10000., 100.]], dtype=torch.float)
        #beta_init = torch.tensor([[100., 100.], [10000., 100.]], dtype=torch.float)
        alpha_init = torch.tensor([[500., 500.], [500., 500.]], dtype=torch.float)
        beta_init = torch.tensor([[500., 500.], [500., 500.]], dtype=torch.float)
        a_alpha = pyro.param("a_alpha_{}".format(k), alpha_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
        a_beta = pyro.param("a_beta_{}".format(k), beta_init[k], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!

        _q_z_a_tau = dist.Beta(a_alpha, a_beta).to_event(1)
        z_a = pyro.sample("z_a", _q_z_a_tau)
        return z_a

    def p_z_MB_tau(self, tau, z_s_tau, dynamicParams):
        # values should not be changed - use the params in "action_transforme" instead!
        a_min = torch.tensor([0., 0.], dtype=torch.float)
        a_max = torch.tensor([1., 1.], dtype=torch.float)
        _p_z_a_tau = dist.Uniform(a_min, a_max).to_event(1)
        z_a = pyro.sample("z_a", _p_z_a_tau)
        return z_a

    def p_z_s_tau(self, tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams):
        mean = z_s_tauMinus1 + self.action_transforme(z_a_tauMinus1)
        cov = self.params["cov_s"]
        _p_z_s_tau = dist.MultivariateNormal(mean, cov)
        z_s = pyro.sample("z_s", _p_z_s_tau)
        return z_s

    def WM_planning_optimizer(self):
        # https://pyro.ai/examples/svi_part_iv.html
        initial_lr = self.params["initial_lr"]
        gamma = self.params["gamma"]  # final learning rate will be gamma * initial_lr
        lrd = gamma ** (1 / self.params["svi_epochs"])
        optim_args = {'lr': initial_lr, 'lrd': lrd}
        return pyro.optim.ClippedAdam(optim_args)

    def WM_planning_loss(self):
        # https://pyro.ai/examples/svi_part_iv.html
        return pyro.infer.TraceEnum_ELBO(num_particles=1)
        # return pyro.infer.RenyiELBO()  # might be better when K>1


class AppraisalsImplementation(Appraisals):
    def d_c_tau(self, tau, z_s_tau, z_LTM, z_PB_posterior, dynamicParams):
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

    def p_z_PB(self, tau, z_s_tau, dynamicParams):
        p_z_Lidar_prior(self.params["lidarParams"])

    def p_z_PB_posterior(self, tau, z_s_tau, z_LTM, dynamicParams):
        position = z_s_tau  # z_s_tau["position"]
        z_Map = z_LTM["z_Map"]
        z_lidar = p_z_Lidar_posterior(position, z_Map, self.params["lidarParams"])
        z_PB_posterior = {}
        z_PB_posterior["z_lidar"] = z_lidar
        return z_PB_posterior

    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau, ltm, dynamicParams):
        position = z_s_tau  # z_s_tau["position"]
        PB_labels = lidar_generate_labels(self.params["lidarParams"], position, ltm.content["map_grid_probabilities"], subsampling=self.params["information_gain_subsampling"])
        return PB_labels

    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau, ltm, dynamicParams):
        position = z_s_tau  # z_s_tau["position"]
        lidar_generate_labels(self.params["lidarParams"], position, ltm.content["map_grid_probabilities"], subsampling=self.params["constraint_subsampling"])
        # Consider incorporating funtionality to focus subsampling in the direction
        # of movement...

    def I_c_tilde(self, tau, d, dynamicParams):
        # approximation to the indicator function used for distances
        # d: the distance to a constraint
        # _I_c_tilde: the approximation of the indicator function
        # here we have used an approximation shifted in d to account for the size (radius) of the robot
        # and to include an extra distance buffer
        _I_c_tilde = logistic(d, self.params["d_min"], self.params["P_z_C_scale"])
        return _I_c_tilde


class ReflectiveAttentionMechanismImplementation(ReflectiveAttentionMechanism):
    def __init__(self, appraisalsImplementation, PlanningImplementation, params):
        self.branchingStateIndexes = []
        super().__init__(appraisalsImplementation, PlanningImplementation, params)

    deliberateAttentionMechanisms = {"Damgaard2022AKS": Damgaard2022AKS} # Progress might not be necessary

    def reset(self):
        self.appraisalsImplementation.reset()
        self.planningImplementation.reset()

    def WM_reflectiveAttentionMechanism(self, t, T, p_z_s_t, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams):
        deliberateAttentionMechanisms_to_evaluate = []
        deliberateAttentionMechanism_ = self.deliberateAttentionMechanisms["Damgaard2022AKS"](self.appraisalsImplementation, ltm, self.params)
        deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanism_)
        deliberateAttentionMechanisms_evaluated = {}
        deliberateAttentionMechanisms_evaluated.update(self.WM_eval_deliberateAttentionMechanisms(t, T, p_z_s_t, deliberateAttentionMechanisms_to_evaluate, p_z_s_Minus_traces, ltm, dynamicParams, parallel_processing=False))
        chosen_plan = deliberateAttentionMechanisms_evaluated["Damgaard2022AKS"]

        return chosen_plan
