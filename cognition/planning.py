from abc import ABC, abstractmethod
from .misc import KL_point_estimate, Lautum_information_estimate, probabilistic_OR_independent, probabilistic_AND_independent, gradient_modifier
from .stateTree import StateTreeBranch, flattenList

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import scope
import numpy as np
import time


from multiprocessing import Pool, active_children
import sys
import dill as pickle
import numba
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # tries to fix the error below:
# multiprocessing.pool.MaybeEncodingError: Error sending result: .... Reason: 'RuntimeError('unable to open shared memory object </torch_5360_3990544354> in read-write mode')'


from .deliberateAttentionMechanism import DeliberateAttentionMechanism, PosteriorEvaluation


class Planning(ABC):
    def __init__(self):
        self.param_store = pyro.get_param_store()
        self.svi_instance = pyro.infer.SVI(model=self.WM_planning_model,
                                           guide=self.WM_planning_guide,
                                           optim=self.WM_planning_optimizer(),
                                           loss=self.WM_planning_loss())

        self.PosteriorEvaluation = PosteriorEvaluation

    def reset(self):
        pass


    def WM_planning_posterior_estimation(self, t, T, p_z_s_t, p_z_s_Minus_traces, deliberateAttentionMechanism, svi_instance, dynamicParams):
        # take svi steps...
        pyro.clear_param_store() 
        losses = []
        for svi_epoch in range(self.params["svi_epochs"]):
            step_loss = svi_instance.step(t, T, p_z_s_t, p_z_s_Minus_traces, deliberateAttentionMechanism, dynamicParams)
            losses.append(step_loss)
            # print("svi_epoch: " + str(svi_epoch) + "    loss: " + str(step_loss), flush=True)


    def WM_planning_posterior(self, t, T, p_z_s_t, p_z_s_Minus_traces, ltm, deliberateAttentionMechanism, dynamicParams):
        _p_z_s_Minus_traces = p_z_s_Minus_traces.copy()

        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        P_z_C_accum = torch.tensor([1.], dtype=torch.float)

        # fixed number of options with varying probabilities
        assignment_probs = pyro.param('assignment_probs', torch.ones(self.params["K"]) / self.params["K"], constraint=constraints.unit_interval)
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})  # "sequential", "parallel"
        z_a_tauPlus, z_s_tauPlus, P_alpha_T = self.__WM_planning_step_posterior(t + 1, T, k, z_s_t, _p_z_s_Minus_traces, P_z_C_accum, ltm, deliberateAttentionMechanism, dynamicParams)
        z_s_tauPlus.insert(0, z_s_t)

        return z_a_tauPlus, z_s_tauPlus, k, P_alpha_T

    def __WM_planning_step_posterior(self, tau, T, k, z_s_tauMinus1, p_z_s_Minus_traces, P_z_C_accum, ltm, deliberateAttentionMechanism, dynamicParams):
        with scope(prefix=str(tau-1)):
            z_a_tauMinus1 = self.q_z_MB_tau(tau-1, z_s_tauMinus1, k, dynamicParams)

        with scope(prefix=str(tau)):
            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        deliberateAttentionMechanism = self.PosteriorEvaluation(self.appraisalsImplementation, ltm, self.params, p_z_g=deliberateAttentionMechanism.p_z_g, desirability_KL_baseline=deliberateAttentionMechanism.desirability_KL_baseline)
        P_alpha_tau, P_z_C_accum =  deliberateAttentionMechanism.calc_appraisal_probs(tau, T, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams)

        if tau >= T:
            P_alpha_tau["constraints accumulated"] = P_z_C_accum

            #for key in P_alpha_tau:
            #    P_alpha_tau[key] = pyro.sample(key+str(T), dist.Bernoulli(P_alpha_tau[key]))

            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus, P_alpha_tau
        else:
            z_a_tauPlus, z_s_tauPlus, P_alpha_T = self.__WM_planning_step_posterior(tau + 1, T, k, z_s_tau, p_z_s_Minus_traces, P_z_C_accum, ltm, deliberateAttentionMechanism, dynamicParams)
            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus, P_alpha_T

    def __WM_planning_prior(self, t, T, p_z_s_t, p_z_s_Minus_traces, dynamicParams):
        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        # fixed number of options with varying probabilities
        z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_prior(t + 1, T, z_s_t, dynamicParams)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus

    def __WM_planning_step_prior(self, tau, T, z_s_tauMinus1, dynamicParams):
        with scope(prefix=str(tau-1)):
            z_a_tauMinus1 = self.p_z_MB_tau(tau-1, z_s_tauMinus1, dynamicParams)

        with scope(prefix=str(tau)):
            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        if tau >= T:
            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus
        else:
            z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_prior(tau + 1, T, z_s_tau, dynamicParams)
            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus

    def WM_planning_model(self, t, T, p_z_s_t, p_z_s_Minus_traces, deliberateAttentionMechanism, dynamicParams):
        _p_z_s_Minus_traces = p_z_s_Minus_traces.copy()

        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        assignment_probs = torch.ones(self.params["K"]) / self.params["K"]
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})  # "sequential", "parallel"
        # k is only used in the guide, but due to Pyro it also needs to be in the model

        P_z_C_accum = torch.tensor([1.], dtype=torch.float)

        # sample planning steps recursively
        z_a_tauPlus, z_s_tauPlus, P_z_d_end, P_z_C_accum = self.__WM_planning_step_model(t + 1, T, k, z_s_t, _p_z_s_Minus_traces, P_z_C_accum, deliberateAttentionMechanism, dynamicParams)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_step_model(self, tau, T, k, z_s_tauMinus1, p_z_s_Minus_traces, P_z_C_accum, deliberateAttentionMechanism, dynamicParams):
        with scope(prefix=str(tau-1)):
            z_a_tauMinus1 = self.p_z_MB_tau(tau-1, z_s_tauMinus1, dynamicParams)

        with scope(prefix=str(tau)):
            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        P_alpha_tau, P_z_C_accum =  deliberateAttentionMechanism.calc_appraisal_probs(tau, T, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams)

        if tau >= T:  # last timestep
            P_alpha_tau["constraints accumulated"] = P_z_C_accum
            if "desirability" in P_alpha_tau:
                P_z_d_end = P_alpha_tau["desirability"]
            else:
                P_z_d_end = None

            #DeliberateAttentionMechanism.p_x_A(self, tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams)
            deliberateAttentionMechanism.p_x_A(tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams)

            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus, P_z_d_end, P_z_C_accum

        else:  # intermidiate timesteps
            z_a_tauPlus, z_s_tauPlus, P_z_d_end, P_z_C_accum = self.__WM_planning_step_model(tau + 1, T, k, z_s_tau, p_z_s_Minus_traces, P_z_C_accum, deliberateAttentionMechanism, dynamicParams)

            P_alpha_tau["constraints accumulated"] = P_z_C_accum

            #DeliberateAttentionMechanism.p_x_A(self, tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams)
            deliberateAttentionMechanism.p_x_A(tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams)

            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus, P_z_d_end, P_z_C_accum
    
    def WM_planning_guide(self, t, T, p_z_s_t, p_z_s_Minus_traces, deliberateAttentionMechanism, dynamicParams):
        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        # fixed number of options with varying probabilities
        assignment_probs = pyro.param('assignment_probs', torch.ones(self.params["K"]) / self.params["K"], constraint=constraints.unit_interval)
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})  # "sequential", "parallel"
        z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(t + 1, T, k, z_s_t, deliberateAttentionMechanism, dynamicParams)
        z_s_tauPlus.insert(0, z_s_t)

        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_step_guide(self, tau, T, k, z_s_tauMinus1, deliberateAttentionMechanism, dynamicParams):
        with scope(prefix=str(tau-1)):
            z_a_tauMinus1 = self.q_z_MB_tau(tau-1, z_s_tauMinus1, k, dynamicParams)

        with scope(prefix=str(tau)):
            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        if tau >= T:
            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus
        else:
            z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(tau + 1, T, k, z_s_tau, deliberateAttentionMechanism, dynamicParams)
            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus

    # ############### Methods that needs to be implemented by the user! ###############
    @abstractmethod
    def q_z_MB_tau(self, tau, z_s_tauMinus1, k, dynamicParams):
        raise NotImplementedError

    @abstractmethod
    def p_z_MB_tau(self, tau, z_s_tau, dynamicParams):
        raise NotImplementedError

    @abstractmethod
    def p_z_s_tau(self, tau, z_s_tauMinus1, z_a_tauMinus1, dynamicParams):
        raise NotImplementedError

    @abstractmethod
    def WM_planning_optimizer(self):
        raise NotImplementedError

    @abstractmethod
    def WM_planning_loss(self):
        raise NotImplementedError







