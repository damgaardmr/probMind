from abc import ABC, abstractmethod
from .misc import KL_point_estimate, Lautum_information_estimate, probabilistic_OR_independent, probabilistic_AND_independent, gradient_modifier

import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.contrib.autoname import scope


class Planning(ABC):

    params = {}
    LTM = {}
    p_z_s_Minus = []

    def __init__(self,
                 K,  # K: number of options/trajectories to consider
                 M,  # M: number of samples from each independent perception in calculation of the information gain
                 N,  # N: number of LTM samples used in calculation of the information gain
                 G,  # G: number of samples from each independent constraint
                 L,  # L: number of past states to consider in progress
                 svi_epochs,
                 optimizer,
                 desirability_scale_factor=1,
                 info_gain_scale_factor=1,
                 progress_scale_factor=1,
                 Lamda_p_min=0.5,
                 consider_impasse=False,
                 information_gradient_multiplier=1.,
                 constraint_gradient_multiplier=1.,
                 loss=None): # loss: the loss function used for SVI

        self.K = K
        self.M = M
        self.N = N
        self.G = G
        self.L = L

        self.svi_epochs = svi_epochs

        self.params["desirability_scale_factor"] = torch.tensor([desirability_scale_factor], dtype=torch.float)
        self.params["progress_scale_factor"] = torch.tensor([progress_scale_factor], dtype=torch.float)
        self.params["info_gain_scale_factor"] = torch.tensor([info_gain_scale_factor], dtype=torch.float)
        self.params["Lamda_p_min"] = Lamda_p_min

        # In case of vanishing gradients try to modify this
        self.information_gradient_multiplier = torch.tensor(information_gradient_multiplier)
        self.constraint_gradient_multiplier = torch.tensor(constraint_gradient_multiplier)
        self.p_z_g = None

        self.consider_impasse = consider_impasse

        # https://pyro.ai/examples/svi_part_iv.html
        if loss is None:
            loss = pyro.infer.TraceEnum_ELBO(num_particles=1)

        self.svi_instance = pyro.infer.SVI(model=self.__WM_planning_model,
                                           guide=self.__WM_planning_guide,
                                           optim=optimizer,
                                           loss=loss)

    def makePlan(self, t, T_delta, p_z_s_t, N_posterior_samples=1, p_z_g=None):
        # T_delta: number of timesteps to predict into to future
        self.T_delta = T_delta
        T = torch.tensor([t + self.T_delta])

        self.p_z_g = p_z_g
        if self.p_z_g is None and self.consider_impasse is False:
            Print("Warning: Goal set consider_impasse is False, thus goals has now effect!")

        # add current state distribution to p_z_s_Minus and maybe delete TOO old state distributions that will not be used anymore!
        self.p_z_s_Minus.append(poutine.trace(p_z_s_t).get_trace())
        self.params["N_old_states"] = len(self.p_z_s_Minus)
        if len(self.p_z_s_Minus) > self.L:
            del self.p_z_s_Minus[0]

        # when introducing goal consider to scale the scale factors for progress and information gain
        # relative to the initial desirability D_KL, i.e. scale with a baseline:
        # self.params["baseline"] = KL_point_estimate(z_s_0_mean, p_z_s_0, P_g)

        # take svi steps...
        pyro.clear_param_store()
        losses = []
        for svi_epoch in range(self.svi_epochs):
            step_loss = self.svi_instance.step(t, T, p_z_s_t, self.p_z_s_Minus)
            losses.append(step_loss)
            # print("svi_epoch: " + str(svi_epoch) + "    loss: " + str(step_loss), flush=True)

        # sample the next action according to the posterior guide
        z_a_tauPlus_samples = []
        z_s_tauPlus_samples = []
        k_samples = []
        for i in range(N_posterior_samples):
            z_a_tauPlus, z_s_tauPlus, k = self.__WM_planning_guide(t, T, p_z_s_t, self.p_z_s_Minus)
            z_a_tauPlus_samples.append(z_a_tauPlus)
            z_s_tauPlus_samples.append(z_s_tauPlus)
            k_samples.append(k)

        return z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples

    def reset(self):
        self.p_z_s_Minus = []
        pyro.clear_param_store()

    def __WM_planning_model(self, t, T, p_z_s_t, p_z_s_Minus):
        _p_z_s_Minus = p_z_s_Minus.copy()

        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        P_z_C_accum = torch.tensor([1.], dtype=torch.float)


        assignment_probs = torch.ones(self.K) / self.K
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})
        # k is only used in the guide, but due to Pyro it also needs to be in the model

        if self.consider_impasse:
            P_impasse = torch.tensor([1.0]) - self.__P_z_p_tau(t, p_z_s_t_trace, _p_z_s_Minus[0], decay=1.0)
        else:
            P_impasse = torch.tensor([1.0])  # do not account for impasse

        # sample planning steps recursively
        z_a_tauPlus, z_s_tauPlus, P_z_d_end = self.__WM_planning_step_model(t + 1, T, k, z_s_t, _p_z_s_Minus, P_z_C_accum, P_impasse)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_guide(self, t, T, p_z_s_t, p_z_s_Minus):
        with scope(prefix=str(t)):
            p_z_s_t_trace = poutine.trace(p_z_s_t).get_trace()
            z_s_t = p_z_s_t_trace.nodes["_RETURN"]["value"]

        # fixed number of options with varying probabilities
        assignment_probs = pyro.param('assignment_probs', torch.ones(self.K) / self.K, constraint=constraints.unit_interval)
        k = pyro.sample('k', dist.Categorical(assignment_probs), infer={"enumerate": "sequential"})
        z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(t + 1, T, k, z_s_t)
        z_s_tauPlus.insert(0, z_s_t)
        return z_a_tauPlus, z_s_tauPlus, k

    def __WM_planning_step_model(self, tau, T, k, z_s_tauMinus1, p_z_s_Minus, P_z_C_accum, P_impasse):
        with scope(prefix=str(tau)):
            z_a_tauMinus1 = self.p_z_MB_tau(z_s_tauMinus1)

            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(z_s_tauMinus1, z_a_tauMinus1)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        # calculate the probability of the state satiesfying the constraints and accummulate that probability
        P_z_C_tau = self.__P_z_c_tau(z_s_tau, z_s_tauMinus1)
        P_z_C_accum = probabilistic_AND_independent([P_z_C_accum, P_z_C_tau])

        if tau >= T:  # last timestep - consider if goal state is reach
            if P_impasse > torch.tensor([0.90]):  # if the probability of being in an impasse is high, then explore
                P_z_d = torch.tensor([0.0])
            else:  # calculate the pseudo probability of being in the goal state
                P_z_d = self.__P_z_d_tau(tau, p_z_s_tau_trace, self.p_z_g)

            if P_z_d < torch.tensor([0.10]):  # if pseudo probability of being close to the goal is small, then explore
                # calculate the (pseudo) probability of the state giving new information
                P_z_i = self.__P_z_i_tau(z_s_tau)

                # calculate the (pseudo) probability of the state yielding progress compared to previous states
                P_z_p = self.__P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus)
            else:  # focus on achieving the goal
                P_z_i = torch.tensor([0.0])
                P_z_p = torch.tensor([0.0])

            with scope(prefix=str(tau)):
                pyro.sample("x_A", self.__p_z_A_tau(P_z_d, P_z_p, P_z_i, P_z_C_accum), obs=torch.tensor([1.], dtype=torch.float))

            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus, P_z_d
        else:  # intermidiate timesteps
            z_a_tauPlus, z_s_tauPlus, P_z_d_end = self.__WM_planning_step_model(tau + 1, T, k, z_s_tau, p_z_s_Minus, P_z_C_accum, P_impasse)

            if P_z_d_end < torch.tensor([0.10]):  # if pseudo probability of being close to the goal is small, then explore
                # calculate the (pseudo) probability of the state giving new information
                P_z_i = self.__P_z_i_tau(z_s_tau)

                # calculate the (pseudo) probability of the state yielding progress compared to previous states
                P_z_p = self.__P_z_p_tau(tau, p_z_s_tau_trace, p_z_s_Minus)
            else:  # focus on achieving the goal
                P_z_i = torch.tensor([0.0])
                P_z_p = torch.tensor([0.0])

            with scope(prefix=str(tau)):
                pyro.sample("x_A", self.__p_z_A_tau(P_z_d_end, P_z_p, P_z_i, P_z_C_accum), obs=torch.tensor([1.], dtype=torch.float))

            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus, P_z_d_end

    def __WM_planning_step_guide(self, tau, T, k, z_s_tauMinus1):
        with scope(prefix=str(tau)):
            z_a_tauMinus1 = self.q_z_MB_tau(z_s_tauMinus1, k)

            p_z_s_tau_trace = poutine.trace(self.p_z_s_tau).get_trace(z_s_tauMinus1, z_a_tauMinus1)
            z_s_tau = p_z_s_tau_trace.nodes["_RETURN"]["value"]

        if tau >= T:
            z_a_tauPlus = [z_a_tauMinus1]
            z_s_tauPlus = [z_s_tau]
            return z_a_tauPlus, z_s_tauPlus
        else:
            z_a_tauPlus, z_s_tauPlus = self.__WM_planning_step_guide(tau + 1, T, k, z_s_tau)
            z_a_tauPlus.insert(0, z_a_tauMinus1)
            z_s_tauPlus.insert(0, z_s_tau)
            return z_a_tauPlus, z_s_tauPlus

    def __P_z_A_tau(self, P_z_d, P_z_p, P_z_i, P_z_c):
        P_z_A1 = probabilistic_OR_independent([P_z_i, P_z_p, P_z_d])  # <-- the order of args might matter!
        P_z_A = probabilistic_AND_independent([P_z_A1, P_z_c])

        return P_z_A

    def __p_z_A_tau(self, P_z_d, P_z_p, P_z_i, P_z_c):
        P_z_A = self.__P_z_A_tau(P_z_d, P_z_p, P_z_i, P_z_c)
        # print("P_z_d: " + "{:.6f}".format(P_z_d.item()) + "  P_z_p: " + "{:.6f}".format(P_z_p.item()) +
        #       "  P_z_i: " + "{:.6f}".format(P_z_i.item()) + "  P_z_c: " + "{:.6f}".format(P_z_c.item()) +
        #       "  P_z_A: " + "{:.6f}".format(P_z_A.item()))
        return dist.Bernoulli(P_z_A)

    def __P_z_d_tau(self, tau, p_z_s_tau_trace, p_z_g):
        if p_z_g is not None:
            # calculate kl-divergence
            with poutine.block():
                p_z_g_trace = poutine.trace(p_z_g).get_trace()
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_g_trace)

            # calculate "pseudo" probability
            P_z_d = torch.exp(-self.params["desirability_scale_factor"] * KL_estimate)
        else:
            P_z_d = torch.tensor(0.0)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_d = dist.Bernoulli(P_z_d)
        # P_z_d = p_z_d.probs

        return P_z_d

    def __P_z_p_tau(self, tau, p_z_s_tau_trace, p_z_s_Minus, decay=None):
        # make some optimization + add the different parameters to the param dict!
        if not isinstance(p_z_s_Minus, pyro.poutine.trace_struct.Trace):  # check if it is only a single trace or a list
            if self.L > len(p_z_s_Minus):
                _L = len(p_z_s_Minus)
            else:
                _L = self.L
            P_z_p_list = []
            for l in range(_L):  # optimize!!!
                idx = len(p_z_s_Minus) - l
                KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus[idx - 1])

                if decay is None:
                    if _L > 1:
                        Lamda_p_l = 1 - (1 - self.params["Lamda_p_min"]) * (_L - l) / _L
                    else:
                        Lamda_p_l = 1

                # P_z_p_list.append(Lamda_p_l * torch.exp(-self.params["progress_scale_factor"] * KL_estimate))  # <-- alternative
                P_z_p_list.append(torch.tensor([1.], dtype=torch.float) - Lamda_p_l * torch.exp(-self.params["progress_scale_factor"] * KL_estimate))

                # Currently the distributions itself is not needed, only its parameters is
                # so no reason to instantiate it
                # p_z_p_l = dist.Bernoulli(P_z_p_list)
                # P_z_p_l = p_z_p_l.probs

            # P_z_p = torch.tensor([1.], dtype=torch.float) - probabilistic_OR_independent(P_z_p_list)  # <-- alternative
            P_z_p = probabilistic_AND_independent(P_z_p_list)
        else:
            KL_estimate = KL_point_estimate(tau, p_z_s_tau_trace, p_z_s_Minus)
            P_z_p = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["progress_scale_factor"] * KL_estimate)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_p = dist.Bernoulli(P_z_p)
        # P_z_p = p_z_p.probs

        return P_z_p

    def __P_z_i_tau(self, z_s_tau):
        with poutine.block():  # nested inference
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True

            # condition all the relevant distributions on the current state sample, z_s_tau:
            def p_z_1_prior():
                return self.p_z_LTM()

            def p_z_2_prior():
                return self.p_z_PB(_z_s_tau)

            def p_z_2_posterior(z_LTM):
                return self.p_z_PB_posterior(_z_s_tau, z_LTM)

            # create subsampling context for LTM and PB and Fetch labels/keys to use as observation sites
            z_2_labels = self.generate_PB_LTM_information_gain_subsampling_context(z_s_tau.detach())  # might contain pyro.sample statements!

            # Calculate the information gain
            information = Lautum_information_estimate(p_z_1_prior, p_z_2_prior, p_z_2_posterior, z_2_labels, M=self.M, N=self.N)

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
            information_max = torch.max(information)
            _P_z_i = torch.tensor([1.], dtype=torch.float) - torch.exp(-self.params["info_gain_scale_factor"] * information_max)

            if _P_z_i.requires_grad:
                _P_z_i.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_i = _P_z_i.detach()

        P_z_i_out = gradient_modifier.apply(z_s_tau, P_z_i, z_s_tau_grad, self.information_gradient_multiplier)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_i = dist.Bernoulli(P_z_i_out)
        # P_z_i_out = p_z_i.probs

        return P_z_i_out

    def __P_z_c_tau(self, z_s_tau, z_s_tauMinus1):
        with poutine.block():  # nested inference
            _z_s_tau = z_s_tau.detach()
            _z_s_tau.requires_grad = True

            # Create subsampling context for LTM and PB
            self.generate_PB_LTM_constraint_subsampling_context(z_s_tau.detach())  # might contain pyro.sample statements!

            for g in range(self.G):
                z_LTM_g = self.p_z_LTM()  # sample long-term memory
                z_PB_posterior_g = self.p_z_PB_posterior(_z_s_tau, z_LTM_g)  # sample from the posterior perceptual buffer
                d_c_tau = self.d_c_tau(_z_s_tau, z_LTM_g, z_PB_posterior_g)  # calculate the constraint distance
                I_c_tau = torch.zeros(len(d_c_tau))
                for h in range(len(d_c_tau)):  # approximate the indicator function
                    I_c_tau[h] = self.I_c_tilde(d_c_tau[h])  # use a smooth approximation to the indicator function to preserve differentiability

                if g == 0:
                    _P_z_c_tau_ = torch.zeros(len(I_c_tau))
                for h in range(len(I_c_tau)):
                    _P_z_c_tau_[h] = _P_z_c_tau_[h] + I_c_tau[h]

            for h in range(len(_P_z_c_tau_)):
                _P_z_c_tau_[h] = _P_z_c_tau_[h] / self.G

                # Currently the distributions themselves are not needed, only their parameters
                # so no reason to instantiate it
                # p_z_c_tau_[h] = dist.Bernoulli(_P_z_c_tau_[h])
                # _P_z_c_tau_[h] = p_z_c_tau_[h].probs

            _P_z_c_tau = probabilistic_AND_independent(_P_z_c_tau_)

            if _P_z_c_tau.requires_grad:
                _P_z_c_tau.backward()
                z_s_tau_grad = _z_s_tau.grad
            else:
                z_s_tau_grad = None
            P_z_c_tau = _P_z_c_tau.detach()

        P_z_c_tau = gradient_modifier.apply(z_s_tau, P_z_c_tau, z_s_tau_grad, self.constraint_gradient_multiplier)

        # Currently the distribution itself is not needed, only its parameters is
        # so no reason to instantiate it
        # p_z_c = dist.Bernoulli(P_z_c_tau)
        # P_z_c_tau = p_z_c.probs

        return P_z_c_tau

    # ############### Methods that needs to be implemented by the user! ###############
    @abstractmethod
    def q_z_MB_tau(self, z_s_tauMinus1, k):
        raise NotImplementedError

    @abstractmethod
    def p_z_MB_tau(self, z_s_tau):
        raise NotImplementedError

    @abstractmethod
    def p_z_s_tau(self, z_s_tauMinus1, z_a_tauMinus1):
        raise NotImplementedError

    @abstractmethod
    def p_z_LTM(self):
        raise NotImplementedError

    @abstractmethod
    def p_z_PB(self, z_s_tau):
        raise NotImplementedError

    @abstractmethod
    def p_z_PB_posterior(self, z_s_tau, z_LTM):
        raise NotImplementedError

    @abstractmethod
    def I_c_tilde(self, d):
        # approximation to the indicator function used for distances
        # d: the distance to a constraint
        # _I_c_tilde: the approximation of the indicator function which should
        # be a in the interval [0;1] and is a smooth monotonically increasing function 
        # symmetric around _I_c_tilde(0) = 0.5
        raise NotImplementedError

    @abstractmethod
    def d_c_tau(self, z_s_tau, z_LTM, z_PB_posterior):
        # returns list of outputs of constraint indicator functions taking the args:
        # z_s_tau, z_LTM, z_PB_posterior
        # That is the function should return a list of H constraints like:
        # [I_c_1(z_s_tau, z_LTM, z_PB_posterior), ... , I_c_H(z_s_tau, z_LTM, z_PB_posterior)]
        raise NotImplementedError

    @abstractmethod
    def generate_PB_LTM_information_gain_subsampling_context(self, z_s_tau):
        # Function that can be used to create a sub-sampling context for use within
        # p_z_PB(...), p_z_PB_posterior(...), and p_z_LTM(...)
        #
        # The function should return a list of labels/keys specifying the sample sites in
        # p_z_PB(...) that should be considered observations in the calculation of
        # information gain in __P_z_i_tau(...)
        raise NotImplementedError

    @abstractmethod
    def generate_PB_LTM_constraint_subsampling_context(self, z_s_tau):
        # Function that can be used to create a sub-sampling context for use within
        # p_z_PB(...), p_z_PB_posterior(...), and p_z_LTM(...)
        raise NotImplementedError

if __name__ == '__main__':
    a = 1
