from abc import ABC, abstractmethod
import torch
from pyro import poutine

from .reflectiveAttentionMechanism import key_mean

class ProbMindAgent(ABC):
    params = {}
    LTM = {}
    p_z_s_Minus_traces = []
    p_z_s_Minus = []

    def __init__(self,
                 reflectiveAttentionMechanismImplementation): # loss: the loss function used for SVI

        self.reflectiveAttentionMechanismImplementation = reflectiveAttentionMechanismImplementation



    def reset(self):
        self.reflectiveAttentionMechanismImplementation.reset()
        self.p_z_s_Minus = []
        self.p_z_s_Minus_traces = []


    def makePlan(self, t, T_delta, p_z_s_t, ltm, p_z_g=None, dynamicParams = {}):
        # T_delta: number of timesteps to predict into to future
        self.t = t
        self.T_delta = T_delta
        self.T_ = torch.tensor([t + self.T_delta])

        # add current state distribution to p_z_s_Minus and maybe delete TOO old state distributions that will not be used anymore!
        self.p_z_s_Minus.append(p_z_s_t)
        self.p_z_s_Minus_traces.append(poutine.trace(p_z_s_t).get_trace())

        chosen_plan = self.reflectiveAttentionMechanismImplementation.WM_reflectiveAttentionMechanism(t, self.T_, p_z_s_t, self.p_z_s_Minus, self.p_z_s_Minus_traces, ltm, p_z_g, dynamicParams)

        #print("avg timings: ")
        #avg_timings = self.reflectiveAttentionMechanismImplementation.get_avg_timings()
        #for key in avg_timings:
        #    print(key + ": " + str(avg_timings[key]))


        self.deliberateAttentionMechanism = chosen_plan["deliberateAttentionMechanism"]
        z_a_tauPlus_samples = chosen_plan["actions"]
        z_s_tauPlus_samples = chosen_plan["states"]
        k_samples = chosen_plan["k_samples"]
        param_store = chosen_plan["param_store"]
        return z_a_tauPlus_samples, z_s_tauPlus_samples, k_samples, param_store