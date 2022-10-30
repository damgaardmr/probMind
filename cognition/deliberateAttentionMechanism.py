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


import dill as pickle



class DeliberateAttentionMechanism(ABC):
    def __init__(self, 
                 appraisalsImplementation,
                 ltm,
                 params,
                 p_z_g=None,
                 desirability_KL_baseline=None,
                 name=None):
        self.ltm = ltm
        if isinstance(p_z_g, (bytes, bytearray)):
            self.p_z_g = p_z_g
        else:
            self.p_z_g = pickle.dumps(p_z_g)
        self.desirability_KL_baseline = desirability_KL_baseline
        self.__name__ = name
        self.params = params
        self.appraisals_ = appraisalsImplementation

    def reset(self):
        pass

    def get_name(self):
        if self.__name__ is None:
            return self.__class__.__name__
        else:
            return self.__name__

    # the two following functions can be overwritten to treat tau = T as a special case!
    # e.g. in a case where we only want to consider desirability of the last predicted state, but constraints for all predicted states!
    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        return self.calc_appraisal_probs_tau(tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams)

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        return self.P_x_A_tau(P_alpha_tau, dynamicParams)

    def p_x_A(self, tau, T, P_alpha_tau, deliberateAttentionMechanism, dynamicParams):
        if tau == T:
            P_x_A = deliberateAttentionMechanism.P_x_A_T(P_alpha_tau, dynamicParams)
        else:
            P_x_A = deliberateAttentionMechanism.P_x_A_tau(P_alpha_tau, dynamicParams)

        with scope(prefix=str(tau)):
            pyro.sample("x_A", dist.Bernoulli(P_x_A), obs=torch.tensor([1.], dtype=torch.float))


    def calc_appraisal_probs(self, tau, T, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        if tau == T:
            return self.calc_appraisal_probs_T(tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams)
        else:
            return self.calc_appraisal_probs_tau(tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams)

    # ############### Methods that needs to be implemented by the user! ###############
    @abstractmethod
    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        # P_alpha_tau = {}
        # P_alpha_tau[appraisal] = self.P_z_d_tau(...)
        # return P_alpha_tau, P_z_C_accum
        raise NotImplementedError

    @abstractmethod
    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        # This method should return the deliberate attention (pseudo) probability, P_z_A. E.g.
        # P_z_A = probabilistic_AND_independent([P_alpha_tau["P_z_d"], P_alpha_tau["P_z_C_accum"]])
        # return P_z_A
        raise NotImplementedError



class AllAppraisals(DeliberateAttentionMechanism):

    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])

        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["desirability"] = torch.tensor([0.0])
            P_alpha_tau["progress"] = torch.tensor([0.0])
            P_alpha_tau["information gain"] = torch.tensor([0.0])
        else:
            P_alpha_tau["desirability"] = self.appraisals_.P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g, self.desirability_KL_baseline, dynamicParams)
            P_alpha_tau["progress"] = self.appraisals_.P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams)        
            P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)

        return P_alpha_tau, P_z_C_accum    

    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
        else:
            P_alpha_tau_ = []
            for key in P_alpha_tau.keys():
                if (key != "constraints") and (key != "constraints accumulated"):
                    P_alpha_tau_.append(P_alpha_tau[key])

            P_z_A1 = probabilistic_OR_independent(P_alpha_tau_)  # <-- the order of args might matter!
            P_z_A = probabilistic_AND_independent([P_z_A1, P_alpha_tau["constraints accumulated"]])

        return P_z_A

class ConstraintAvoidance(DeliberateAttentionMechanism):
    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])

        return P_alpha_tau, P_z_C_accum

    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        P_z_A = P_alpha_tau["constraints accumulated"]

        return P_z_A

class StateReach(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau


    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])
        
        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["desirability"] = torch.tensor([0.0])
        else:
            P_alpha_tau["desirability"] = self.appraisals_.P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g, self.desirability_KL_baseline, dynamicParams)

        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
        else:
            P_z_A = probabilistic_AND_independent([P_alpha_tau["desirability"], P_alpha_tau["constraints accumulated"]])

        return P_z_A

class StateReachWithProgress(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau

    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])
        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["desirability"] = torch.tensor([0.0])
            P_alpha_tau["progress"] = torch.tensor([0.0])
        else:
            P_alpha_tau["desirability"] = self.appraisals_.P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g, self.desirability_KL_baseline, dynamicParams)
            P_alpha_tau["progress"] = self.appraisals_.P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams)

        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
            #P_z_A = probabilistic_AND_independent([P_alpha_tau["progress"], P_alpha_tau["constraints accumulated"]])  # progress should always be possible
        else:
            P_z_A_ = probabilistic_OR_independent([P_alpha_tau["desirability"], P_alpha_tau["progress"]])
            P_z_A = probabilistic_AND_independent([P_z_A_, P_alpha_tau["constraints accumulated"]])
            #P_z_A = probabilistic_AND_independent([P_alpha_tau["information gain"], P_alpha_tau["desirability"], P_alpha_tau["constraints accumulated"]])

        return P_z_A

class StateReachWithExplore(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau


    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])

        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["desirability"] = torch.tensor([0.0])
            P_alpha_tau["information gain"] = torch.tensor([0.0])
        else:
            P_alpha_tau["desirability"] = self.appraisals_.P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g, self.desirability_KL_baseline, dynamicParams)
            P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)

        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
        else:
            P_z_A_ = probabilistic_OR_independent([P_alpha_tau["desirability"], P_alpha_tau["information gain"]])
            P_z_A = probabilistic_AND_independent([P_z_A_, P_alpha_tau["constraints accumulated"]])
            #P_z_A = probabilistic_AND_independent([P_alpha_tau["information gain"], P_alpha_tau["desirability"], P_alpha_tau["constraints accumulated"]])

        return P_z_A

class Explore(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau

    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])
        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["information gain"] = torch.tensor([0.0])
        else:
            P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)   


        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
        else:
            P_z_A = probabilistic_AND_independent([P_alpha_tau["information gain"], P_alpha_tau["constraints accumulated"]])

        return P_z_A

class ExploreWithoutConstraint(DeliberateAttentionMechanism):
    def calc_appraisal_probs_tau(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_z_C_accum = torch.tensor([0.0])
        return P_alpha_tau, P_z_C_accum

    def P_x_A_tau(self, P_alpha_tau, dynamicParams):
        #P_z_A = P_alpha_tau["information gain"]
        P_z_A = torch.tensor([0.0])
        return P_z_A

    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}
        P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)
        P_z_C_accum = torch.tensor([0.0])
   
        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        P_z_A = P_alpha_tau["information gain"]
        return P_z_A

class ExploreWithProgress(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau

    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}       
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])
        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["progress"] = torch.tensor([0.0])
            P_alpha_tau["information gain"] = torch.tensor([0.0])
        else:
            P_alpha_tau["information gain"] = self.appraisals_.P_z_i_tau(tau, z_s_tau, self.ltm, dynamicParams)
            P_alpha_tau["progress"] = self.appraisals_.P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams) 

        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
            #P_z_A = probabilistic_AND_independent([P_alpha_tau["progress"], P_alpha_tau["constraints accumulated"]])  # progress should always be possible
        else:
            P_z_A_ = probabilistic_OR_independent([P_alpha_tau["information gain"], P_alpha_tau["progress"]])
            P_z_A = probabilistic_AND_independent([P_z_A_, P_alpha_tau["constraints accumulated"]])

        return P_z_A

class Progress(ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau

    def calc_appraisal_probs_T(self, tau, z_s_tau, z_s_tauMinus1, p_z_s_tau_trace, P_z_C_accum, p_z_s_Minus_traces, dynamicParams):
        P_alpha_tau = {}        
        P_alpha_tau["constraints"] = self.appraisals_.P_z_c_tau(tau, z_s_tau, z_s_tauMinus1, self.ltm, dynamicParams)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_alpha_tau["constraints"]])
        if P_z_C_accum < torch.tensor([1.0]):
            P_alpha_tau["progress"] = torch.tensor([0.0])
        else:
            P_alpha_tau["progress"] = self.appraisals_.P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams) 
        return P_alpha_tau, P_z_C_accum

    def P_x_A_T(self, P_alpha_tau, dynamicParams):
        if P_alpha_tau["constraints accumulated"] < torch.tensor([1.0]):
            P_z_A = P_alpha_tau["constraints accumulated"]
        else:
            P_z_A = probabilistic_AND_independent([P_alpha_tau["progress"], P_alpha_tau["constraints accumulated"]])

        return P_z_A

class PosteriorEvaluation(AllAppraisals, ConstraintAvoidance):
    calc_appraisal_probs_tau = ConstraintAvoidance.calc_appraisal_probs_tau
    P_x_A_tau = ConstraintAvoidance.P_x_A_tau

    calc_appraisal_probs_T = AllAppraisals.calc_appraisal_probs_tau
    P_x_A_T = AllAppraisals.P_x_A_tau