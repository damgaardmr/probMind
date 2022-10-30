import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import scope

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
from cognition.ltm import LTM
from cognition.appraisals import Appraisals
from cognition.reflectiveAttentionMechanism import ReflectiveAttentionMechanism
from cognition.deliberateAttentionMechanism import DeliberateAttentionMechanism
from cognition.misc import probabilistic_OR_independent, probabilistic_AND_independent



class Damgaard2022MR(DeliberateAttentionMechanism):
    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["P_z_o"] = self.cost(tau, z_s_tau, dynamicParams["z_goal"])

        if dynamicParams["msgs_received"] != None:
            for ID in range(len(dynamicParams["msgs_received"])):
                if str(ID) in z_s_tau:
                    min_dist = self.params["radius"] + dynamicParams["msgs_received"][ID][8]
                    P_z_c_ = self.constraint(z_s_tau["own_pose"], z_s_tau[str(ID)], min_dist)
                    P_alpha_tau["P_z_c_{}".format(ID)] = P_z_c_

        P_z_C_accum = torch.tensor([0.0])

        return P_alpha_tau, P_z_C_accum

    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        # not used since _DeliberateAttentionMechanism__p_x_A() is overwritten
        return None

    # ################### overwrite parent "private" class methods ####################
    def p_x_A(self, tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams):
        # The planning logic in the paper related to this simulation is not quit the same as for the planning idiom. Therefore we overwrite it.
        # Here we observe the optimality variable as well as the constraint variable at each time step
        # However we could easily have used to logic of the planning idiom instead! 

        with scope(prefix=str(tau)):
            if dynamicParams["msgs_received"] != None:
                for ID in range(len(dynamicParams["msgs_received"])):
                    if "P_z_c_{}".format(ID) in P_alpha_tau:
                        pyro.sample("c_{}_{}_{}".format(self.params["ID"], ID,tau), dist.Bernoulli(P_alpha_tau["P_z_c_{}".format(ID)]), obs=torch.tensor([0.],dtype=torch.float)) # constraint!

            pyro.sample("x_o_{}".format(self.params["ID"]), dist.Bernoulli(P_alpha_tau["P_z_o"]), obs=torch.tensor([1.], dtype=torch.float))

    #@torch.jit.script
    def constraint(self, own_pose, others_pose, min_dist):
        dist = torch.dist(own_pose[torch.tensor([0, 1])], others_pose[torch.tensor([0, 1])], p=2)
        if dist<= min_dist:
            return torch.tensor([1.],dtype=torch.float)
        else:
            dist = dist - min_dist
            return torch.exp(-dist*self.params["constraint_scale_factor"]) # the higher constant the closer we allow the robots

    def cost(self, tau, z_s_tau, z_goal):
        pose = z_s_tau["own_pose"]
        C = torch.dist(pose[[0,1]], z_goal.index_select(0, torch.tensor([0, 1])),p=2)
        # if C > 2.0:  # this allows us to tune the "desirability_scale_factor" and learning rates for goals a specific distance away from the robot
        #     goal_vector = self.z_goal.index_select(0, torch.tensor([0, 1])).detach() - pose[[0,1]].detach()
        #     goal_vector_normalized = torch.nn.functional.normalize(goal_vector, dim=0).detach()
        #     goal_tmp = pose[[0,1]].detach() + 2.0*goal_vector_normalized.detach()
        #     C = torch.dist(pose[[0,1]], goal_tmp,p=2)

        P_z_o = torch.exp(-self.params["desirability_scale_factor"] * C)
        return P_z_o


class LTMImplementation(LTM):
    def __init__(self):
        super().__init__()

    def p_z_LTM(self, dynamicParams):
        return None


class PlanningImplementation(Planning):
    def __init__(self, 
                 params,
                 appraisalsImplementation):
        self.params = params

        #standard_diviation = torch.tensor(params["movement_3_sigma"] / 3)  # 99.7% of samples within a circle of 25 cm
        #variance = standard_diviation * standard_diviation
        #self.params["cov_s"] = variance * torch.eye(2)
        #self.params["a_support"] = torch.tensor(params["a_support"], dtype=torch.float)  # 1 m in each direction

        self.appraisalsImplementation = appraisalsImplementation

        super().__init__()

        self.PosteriorEvaluation = Damgaard2022MR


    # ################### Abstract methods of the class Planning ####################
    def q_z_MB_tau(self, tau, z_s_tauMinus1, k, dynamicParams):
        z_a = {}

        if tau == dynamicParams["T"]:  # this is fixed while we are planning
            z_a["own_action"] = dynamicParams["z_a_t_old"]
        else:
            alpha_init = torch.tensor(self.params["alpha_init"], dtype=torch.float) # small preference for going forward initially 
            beta_init = torch.tensor(self.params["beta_init"], dtype=torch.float)
            # parameters are not automatically scoped!
            a_alpha_trans = pyro.param(str(tau)+"/"+"a_alpha_trans_{}_{}".format(self.params["ID"], k), alpha_init[0], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_alpha_rot = pyro.param(str(tau)+"/"+"a_alpha_rot_{}_{}".format(self.params["ID"], k), alpha_init[1], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_beta_trans = pyro.param(str(tau)+"/"+"a_beta_trans_{}_{}".format(self.params["ID"], k), beta_init[0], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!
            a_beta_rot = pyro.param(str(tau)+"/"+"a_beta_rot_{}_{}".format(self.params["ID"], k), beta_init[1], constraint=constraints.positive)  # alpha,beta = 1 gives uniform!

            # a_beta_trans = torch.exp(a_beta_trans/1000/5)  # can potentially make the actions more aggressive!
            # a_alpha_trans = a_alpha_trans*a_alpha_trans*a_alpha_trans  # can potentially make the actions more aggressive!

            a_alpha = torch.stack((a_alpha_trans, a_alpha_rot), 0)
            a_beta = torch.stack((a_beta_trans, a_beta_rot), 0)

            _q_z_a_tau = dist.Beta(a_alpha, a_beta).to_event(1)
            z_a["own_action"] = pyro.sample("z_a_{}".format(self.params["ID"]), _q_z_a_tau)

        return z_a

    def p_z_MB_tau(self, tau, z_s_tau, dynamicParams):
        z_a = {}
        if tau == dynamicParams["T"]:  # this is fixed while we are planning
            z_a["own_action"] = dynamicParams["z_a_t_old"]
        else:
            # values should not be changed - use the params in "action_transforme" instead!
            a_min = torch.tensor([0., 0.], dtype=torch.float)
            a_max = torch.tensor([1., 1.], dtype=torch.float)
            _p_z_a_tau = dist.Uniform(a_min, a_max).to_event(1)
            z_a["own_action"] = pyro.sample("z_a_{}".format(self.params["ID"]), _p_z_a_tau)

        with poutine.block():  # we only need the samples!
            with torch.no_grad():
                if dynamicParams["msgs_received"] != None:
                    for ID in range(len(dynamicParams["msgs_received"])):
                        if tau == dynamicParams["T"]:  # this is fixed while we are planning
                            z_a[str(ID)] = dynamicParams["msgs_received"][ID][7]
                        elif tau == dynamicParams["T"] + 1:  # this is fixed while we are planning
                            z_a[str(ID)] = dynamicParams["msgs_received"][ID][2]
                        else:
                            if tau-(dynamicParams["T"]+2) <= len(dynamicParams["msgs_received"][ID][3]):
                                a_alpha_ = dynamicParams["msgs_received"][ID][3][tau-(dynamicParams["T"]+2)]
                                a_beta_ = dynamicParams["msgs_received"][ID][4][tau-(dynamicParams["T"]+2)]
                                z_a[str(ID)] = pyro.sample("z_a_{}_{}".format(self.params["ID"], ID), dist.Beta(a_alpha_, a_beta_).to_event(1)).detach()
        return z_a

    def p_z_s_tau(self, tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams):
        z_s = {}
        if tau == dynamicParams["T"]:  # only propagate time until the end of planning
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.params["ID"]), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], dynamicParams["t_avg_planning"]))
        if tau == dynamicParams["T"] + 1:  # propagate time left until next timestep!
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.params["ID"]), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], dynamicParams["t_left_from_predicted_pose"]))
        else:  # propagate time a whole timestep
            z_s["own_pose"] = pyro.sample("z_s_{}".format(self.params["ID"]), self.F(z_s_tauMinus1["own_pose"], z_a_tauMinus1["own_action"], dynamicParams["t_delta"]))


        with poutine.block():  # we only need the samples!
            with torch.no_grad():
                if dynamicParams["msgs_received"] != None:
                    for ID in range(len(dynamicParams["msgs_received"])):
                        if str(ID) in z_a_tauMinus1:  # if the function is called from the guide "z_a_tauMinus1" does not exist!
                            if tau == dynamicParams["T"]:  # only propagate time until the end of planning
                                t_avg_planning = dynamicParams["msgs_received"][ID][5]
                                z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.params["ID"],ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], dynamicParams["t_avg_planning"])).detach()
                            elif tau == dynamicParams["T"] + 1:  # propagate time left until next timestep!
                                t_left_from_predicted_pose = dynamicParams["msgs_received"][ID][6]
                                z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.params["ID"],ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], dynamicParams["t_left_from_predicted_pose"])).detach()
                            else:  # propagate time a whole timestep
                                if tau-(dynamicParams["T"]+2) <= len(dynamicParams["msgs_received"][ID][3]):
                                    z_s[str(ID)] = pyro.sample("z_s_{}_{}".format(self.params["ID"],ID), self.F(z_s_tauMinus1[str(ID)], z_a_tauMinus1[str(ID)], dynamicParams["t_delta"])).detach()

        return z_s

    def WM_planning_optimizer(self):
        def per_param_lr(module_name, param_name):
            if "a_alpha_rot_" in param_name:
                return {"lr": self.params["lr_a_alpha_rot"]}
            elif "a_beta_rot_" in param_name:
                return {"lr": self.params["lr_a_beta_rot"]}
            elif "a_alpha_trans_" in param_name:
                return {"lr": self.params["lr_a_alpha_trans"]}
            elif "a_beta_trans_" in param_name:
                return {"lr": self.params["lr_a_beta_trans"]}
            else:
                return {"lr": self.params["lr"]}
        optimizer = pyro.optim.Adam(per_param_lr)

        return optimizer

    def WM_planning_loss(self):
        # https://pyro.ai/examples/svi_part_iv.html
        return pyro.infer.TraceEnum_ELBO(num_particles=1)
        # return pyro.infer.RenyiELBO()  # might be better when K>1

    # ################### other methods ####################
    # @torch.jit.script
    def action_transforme(self, z_a_t):
        # scaling parameters for the action
        a_support = torch.tensor(self.params["a_support"], dtype=torch.float) # [m/s,rad/s] turtlebot3 burger
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

        M = t_delta*torch.tensor(self.params["model_error"],dtype=torch.float)
        return dist.Uniform(mean-M, mean+M).to_event(1)


class AppraisalsImplementation(Appraisals):
    def d_c_tau(self, tau, z_s_tau, z_LTM, z_PB_posterior, dynamicParams):
        return None

    def p_z_PB(self, tau, z_s_tau, dynamicParams):
        return None #p_z_Lidar_prior(self.params["lidarParams"])

    def p_z_PB(self, tau, z_s_tau, dynamicParams):
        return None

    def p_z_PB_posterior(self, tau, z_s_tau, z_LTM, dynamicParams):
        return None

    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau, dynamicParams):
        return []

    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau, dynamicParams):
        return []

    def I_c_tilde(self, tau, d, dynamicParams):
        return None


class ReflectiveAttentionMechanismImplementation(ReflectiveAttentionMechanism):
    def __init__(self, appraisalsImplementation, PlanningImplementation, params):
        self.branchingStateIndexes = []
        super().__init__(appraisalsImplementation, PlanningImplementation, params)

    deliberateAttentionMechanisms = {"Damgaard2022MR": Damgaard2022MR} # Progress might not be necessary

    def reset(self):
        self.appraisalsImplementation.reset()
        self.planningImplementation.reset()

    def WM_reflectiveAttentionMechanism(self, t, T, p_z_s_t, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams):
        deliberateAttentionMechanisms_to_evaluate = []
        deliberateAttentionMechanism_ = self.deliberateAttentionMechanisms["Damgaard2022MR"](self.appraisalsImplementation, ltm, self.params)
        deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanism_)
        deliberateAttentionMechanisms_evaluated = {}
        deliberateAttentionMechanisms_evaluated.update(self.WM_eval_deliberateAttentionMechanisms(t, T, p_z_s_t, deliberateAttentionMechanisms_to_evaluate, p_z_s_Minus_traces, ltm, dynamicParams, parallel_processing=False))
        chosen_plan = deliberateAttentionMechanisms_evaluated["Damgaard2022MR"]

        return chosen_plan
