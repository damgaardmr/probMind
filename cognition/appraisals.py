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

class Appraisals(ABC):

    def __init__(self, params):
        self.params = params

    def reset(self):
        pass

    def P_z_d_tau(self, tau, p_z_s_tau_trace, _p_z_g, baseline, dynamicParams):
        p_z_g = pickle.loads(_p_z_g)
        if p_z_g is not None:
            # calculate kl-divergence
            with poutine.block():
                p_z_g_trace = poutine.trace(p_z_g).get_trace()
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_g_trace)

            if baseline <= torch.finfo(torch.float32).eps: # avoid division by zero
                baseline = torch.finfo(torch.float32).eps


            if KL_estimate > baseline:
                P_z_d = torch.tensor(0.0)
            else:
                KL_estimate = KL_estimate / baseline
                P_z_d = torch.exp(-self.params["desirability_scale_factor"] * KL_estimate)

        else:
            P_z_d = torch.tensor(0.0)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_d = dist.Bernoulli(P_z_d)
        # P_z_d = p_z_d.probs

        return P_z_d

    def P_z_p_tau(self, tau, p_z_s_tau_trace, p_z_s_Minus_traces, dynamicParams, Lamda_p_l=None, return_least_progress_index=False):
        scale_factor = self.params["progress_scale_factor"]

        idx_2 = len(p_z_s_Minus_traces)-1
        idx_1 = idx_2-self.params["L"]
        p_z_s_Minus_traces_ = p_z_s_Minus_traces[idx_1:idx_2]
        # make some optimization + add the different parameters to the param dict!
        if not isinstance(p_z_s_Minus_traces_, pyro.poutine.trace_struct.Trace):  # check if it is only a single trace or a list
            if Lamda_p_l is None:
                cal_Lamda_p_l = True
            else:
                cal_Lamda_p_l = False

            _L = len(p_z_s_Minus_traces_)
            P_z_p_list = []
            for l in range(_L):  # optimize!!!
                idx = _L - l
                KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus_traces_[idx - 1])

                if cal_Lamda_p_l:
                    if _L > 1:
                        Lamda_p_l = 1 - (1 - self.params["Lamda_p_min"]) * (_L - l) / _L
                    elif _L == 1:
                        Lamda_p_l = 1
                else:
                    Lamda_p_l = Lamda_p_l

                # P_z_p_list.append(Lamda_p_l * torch.exp(-self.params["progress_scale_factor"] * KL_estimate))  # <-- alternative
                P_z_p_list.append(torch.tensor([1.], dtype=torch.float) - Lamda_p_l * torch.exp(-scale_factor * KL_estimate))

                # Currently the distributions itself is not needed, only its parameters is
                # so no reason to instantiate it
                # p_z_p_l = dist.Bernoulli(P_z_p_list)
                # P_z_p_l = p_z_p_l.probs

            # P_z_p = torch.tensor([1.], dtype=torch.float) - probabilistic_OR_independent(P_z_p_list)  # <-- alternative
            P_z_p = probabilistic_AND_independent(P_z_p_list)

            if return_least_progress_index:
                if P_z_p_list: # not empty
                    P_z_p_list_ = torch.stack(P_z_p_list)
                    least_progress_index = _L - torch.argmin(P_z_p_list_).item()
                else:
                    least_progress_index = 0
                return P_z_p, least_progress_index
        else:
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus_traces_)
            P_z_p = torch.tensor([1.], dtype=torch.float) - torch.exp(-scale_factor * KL_estimate)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_p = dist.Bernoulli(P_z_p)
        # P_z_p = p_z_p.probs

        return P_z_p

    #def P_z_i_tau(self, tau, z_s_tau):
    #    return self.__P_z_i_tau(tau, z_s_tau)

    def P_z_i_tau(self, tau, z_s_tau, ltm, dynamicParams, mode="max"):
        with poutine.block():  # nested inference
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True

            # condition all the relevant distributions on the current state sample, z_s_tau:
            def p_z_1_prior():
                return ltm.p_z_LTM(dynamicParams)

            def p_z_2_prior():
                return self.p_z_PB(tau, _z_s_tau, dynamicParams)

            def p_z_2_posterior(z_LTM):
                return self.p_z_PB_posterior(tau, _z_s_tau, z_LTM, dynamicParams)

            # create subsampling context for LTM and PB and Fetch labels/keys to use as observation sites
            z_2_labels = self.generate_PB_LTM_information_gain_subsampling_context(z_s_tau.detach(), ltm, dynamicParams)  # might contain pyro.sample statements!

            # Calculate the information gain
            information = Lautum_information_estimate(p_z_1_prior, p_z_2_prior, p_z_2_posterior, z_2_labels, M=self.params["M"], N=self.params["N"])

            # if len(PB_labels)>1 it might be possible that it is possible to obtain information in different directions, i.e.
            # the gradients for each of these observations might be working against each other. Therefore, we only seek in the direction
            # with most information:
            # P_z_i_j = torch.zeros(len(information))
            # for j in range(len(information)):
            #     P_z_i_j[j] = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["info_gain_scale_factor"] * information[j])
            #     # Currently the distributions themselves are not needed, only their parameters so no reason to instantiate them
            #     # p_z_i_j[j] = dist.Bernoulli(P_z_i_j[j])
            #     # P_z_i_j[j] = p_z_i_j[h].probs
            # _P_z_i = torch.max(P_z_i_j)

            # Since we are not using the distributions we can optimize out the intermidiate calculations
            if mode == "max":
                information_max = torch.max(information)
                _P_z_i = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["info_gain_scale_factor"] * information_max)
            elif mode == "or":
                information_max = probabilistic_OR_independent(information)
                _P_z_i = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["info_gain_scale_factor"] * information_max)
            else:
                print("Error information mode not known")

            if _P_z_i.requires_grad:
                _P_z_i.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_i = _P_z_i.detach()

        P_z_i_out = gradient_modifier.apply(z_s_tau, P_z_i, z_s_tau_grad, self.params["information_gradient_multiplier"])


        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_i = dist.Bernoulli(P_z_i_out)
        # P_z_i_out = p_z_i.probs

        return P_z_i_out

    def P_z_c_tau(self, tau, z_s_tau, z_s_tauMinus1, ltm, dynamicParams, mode="min"):
        with poutine.block():  # nested inference
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True

            # Create subsampling context for LTM and PB
            self.generate_PB_LTM_constraint_subsampling_context(z_s_tau.detach(), ltm, dynamicParams)  # might contain pyro.sample statements!

            _P_z_c_tau_ = []

            for g in range(self.params["G"]):
                z_LTM_g = ltm.p_z_LTM(dynamicParams)  # sample long-term memory
                z_PB_posterior_g = self.p_z_PB_posterior(tau, _z_s_tau, z_LTM_g, dynamicParams)  # sample from the posterior perceptual buffer
                d_c_tau = self.d_c_tau(tau, _z_s_tau, z_LTM_g, z_PB_posterior_g, dynamicParams)  # calculate the constraint distance
                I_c_tau = torch.zeros(len(d_c_tau))
                for h in range(len(d_c_tau)):  # approximate the indicator function
                    I_c_tau[h] = self.I_c_tilde(tau, d_c_tau[h], dynamicParams)  # use a smooth approximation to the indicator function to preserve differentiability

                if g == 0:
                    _P_z_c_tau_ = torch.zeros(len(I_c_tau))
                for h in range(len(I_c_tau)):
                    _P_z_c_tau_[h] = _P_z_c_tau_[h] + I_c_tau[h]

            for h in range(len(_P_z_c_tau_)):
                _P_z_c_tau_[h] = _P_z_c_tau_[h] / self.params["G"]

                # Currently the distributions themselves are not needed, only their parameters
                # so no reason to instantiate it
                # p_z_c_tau_[h] = dist.Bernoulli(_P_z_c_tau_[h])
                # _P_z_c_tau_[h] = p_z_c_tau_[h].probs

            #_P_z_c_tau = probabilistic_AND_independent(_P_z_c_tau_)

            # Since we are not using the distributions we can optimize out the intermidiate calculations
            if mode == "min":
                _P_z_c_tau = torch.min(_P_z_c_tau_)
            elif mode == "or":
                _P_z_c_tau = probabilistic_AND_independent(_P_z_c_tau_)
            else:
                print("Error information mode not known")


            if _P_z_c_tau.requires_grad:
                _P_z_c_tau.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_c_tau = _P_z_c_tau.detach()

        P_z_c_tau = gradient_modifier.apply(z_s_tau, P_z_c_tau, z_s_tau_grad, self.params["constraint_gradient_multiplier"])

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_c = dist.Bernoulli(P_z_c_tau)
        # P_z_c_tau = p_z_c.probs

        return P_z_c_tau

    # ############### Methods that needs to be implemented by the user! ###############
    @abstractmethod
    def p_z_PB(self, tau, z_s_tau, dynamicParams):
        raise NotImplementedError

    @abstractmethod
    def p_z_PB_posterior(self, tau, z_s_tau, z_LTM, dynamicParams):
        raise NotImplementedError

    @abstractmethod
    def I_c_tilde(self, tau, d, dynamicParams):
        # approximation to the indicator function used for distances
        # d: the distance to a constraint
        # _I_c_tilde: the approximation of the indicator function which should
        # be a in the interval [0;1] and is a smooth monotonically increasing function 
        # symmetric around _I_c_tilde(0) = 0.5
        raise NotImplementedError

    @abstractmethod
    def d_c_tau(self, tau, z_s_tau, z_LTM, z_PB_posterior, dynamicParams):
        # returns list of outputs of constraint indicator functions taking the args:
        # z_s_tau, z_LTM, z_PB_posterior
        # That is the function should return a list of H constraints like:
        # [I_c_1(z_s_tau, z_LTM, z_PB_posterior), ... , I_c_H(z_s_tau, z_LTM, z_PB_posterior)]
        raise NotImplementedError

    @abstractmethod
    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau, dynamicParams):
        # Function that can be used to create a sub-sampling context for use within
        # p_z_PB(...), p_z_PB_posterior(...), and p_z_LTM(...)
        #
        # The function should return a list of labels/keys specifying the sample sites in
        # p_z_PB(...) that should be considered observations in the calculation of
        # information gain in __P_z_i_tau(...)
        raise NotImplementedError

    @abstractmethod
    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau, dynamicParams):
        # Function that can be used to create a sub-sampling context for use within
        # p_z_PB(...), p_z_PB_posterior(...), and p_z_LTM(...)
        raise NotImplementedError








