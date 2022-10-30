from abc import ABC, abstractmethod

import torch
import numpy as np
import pyro
from pyro.contrib.autoname import scope
import time

from multiprocessing import Pool #, active_children
import dill as pickle
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')  # tries to fix the error below:
# multiprocessing.pool.MaybeEncodingError: Error sending result: .... Reason: 'RuntimeError('unable to open shared memory object </torch_5360_3990544354> in read-write mode')'


from .deliberateAttentionMechanism import DeliberateAttentionMechanism, AllAppraisals, ConstraintAvoidance, StateReach, StateReachWithProgress, StateReachWithExplore, Explore, ExploreWithProgress, Progress, PosteriorEvaluation


def key_mean(_list, key):
    mean = torch.tensor(0.0)

    for item in _list:
        mean = mean + item[key]

    mean = mean/len(_list)
    return mean

def WM_eval_deliberateAttentionMechanism(arg): # defined outsite class definition to make multiprocessing possible...
    tic1 = time.time()
    t, T, _p_z_s_t, deliberateAttentionMechanism_to_evaluate, planningImplementation, p_z_s_Minus_traces, _ltm, N_posterior_samples, pyro_enable_validation, dynamicParams = arg
    pyro.enable_validation(pyro_enable_validation)

    p_z_s_t = pickle.loads(_p_z_s_t)
    ltm = pickle.loads(_ltm)

    with planningImplementation.param_store.scope() as param_scope_:
        # first we estimate the posterior with the given deliberate attention mechanism (i.e. using a subset of the appraisals)
        planningImplementation.WM_planning_posterior_estimation(t, T, p_z_s_t, p_z_s_Minus_traces, deliberateAttentionMechanism_to_evaluate, planningImplementation.svi_instance, dynamicParams)

        tic2 = time.time()
        with torch.no_grad(): # we do not need the grads for this part
            # then sample all of the appraisals and actions given the posterior estimate
            z_a_tauPlus_samples = []
            z_s_tauPlus_samples = []
            k_samples = []
            P_alpha_T_samples = []

            for i in range(N_posterior_samples):
                z_a_tauPlus, z_s_tauPlus, k, P_alpha_T = planningImplementation.WM_planning_posterior(t, T, p_z_s_t, p_z_s_Minus_traces, ltm, deliberateAttentionMechanism_to_evaluate, dynamicParams)
                z_a_tauPlus_ = []
                for item in z_a_tauPlus:
                    #z_a_tauPlus_.append(item.detach())
                    z_a_tauPlus_.append(item)

                z_s_tauPlus_ = []
                for item in z_s_tauPlus:
                    #z_s_tauPlus_.append(item.detach())
                    z_s_tauPlus_.append(item)

                P_alpha_T_ = {}
                for key in P_alpha_T:
                    #P_alpha_T_[key] = P_alpha_T[key].detach()
                    P_alpha_T_[key] = P_alpha_T[key]

                z_a_tauPlus_samples.append(z_a_tauPlus_)
                z_s_tauPlus_samples.append(z_s_tauPlus_)
                k_samples.append(k)
                P_alpha_T_samples.append(P_alpha_T_)
    
            P_alpha_T_expectations = {}
            for key in P_alpha_T_samples[0].keys():
                P_alpha_T_expectations[key] = key_mean(P_alpha_T_samples,key)

            toc = time.time()
            #print(planningImplementation.param_store.keys())
            #print("{}/a_alpha_trans_{}_{}".format(t, planningImplementation.params["ID"], k))
            #print(planningImplementation.param_store["{}/a_alpha_trans_{}_{}".format(t+1, planningImplementation.params["ID"], k)])

            evaluation = {}
            evaluation["param_store"] = {}
            for key in planningImplementation.param_store.keys():
                evaluation["param_store"][key] = planningImplementation.param_store[key]
            evaluation["deliberateAttentionMechanism"] = deliberateAttentionMechanism_to_evaluate.get_name()
            evaluation["actions"] = z_a_tauPlus_samples
            evaluation["states"] = z_s_tauPlus_samples
            evaluation["k_samples"] = k_samples
            #evaluation["P_alpha_T_samples"] = P_alpha_T_samples
            evaluation["P_alpha_T_expectations"] = P_alpha_T_expectations
            evaluation["infer_action_posterior_est"] = tic2-tic1
            evaluation["appraisals_estimation"] = toc-tic2

    return evaluation


class ReflectiveAttentionMechanism(ABC):
    def __init__(self, appraisalsImplementation, PlanningImplementation, params, n_workers=1):
        self.appraisalsImplementation = appraisalsImplementation
        self.planningImplementation = PlanningImplementation
        self.params = params
        self.params["n_workers"] = n_workers
        self.pool = None
        self.timings = {}
        self.timings["infer_action_posterior_est"] = {}
        self.timings["appraisals_estimation"] = {}
        self.pyro_enable_validation = False  # <-- if you are having trouble with pyro set this to "True"!!!

    @property
    @abstractmethod
    def deliberateAttentionMechanisms(self):
        raise NotImplementedError

    def reset(self):
        self.appraisalsImplementation.reset()
        self.planningImplementation.reset()
        self.timings = {}
        self.timings["infer_action_posterior_est"] = {}
        self.timings["appraisals_estimation"] = {}

    def get_avg_timings(self, N_samples_to_exclude = 3):
        avg_timings = {}
        for key1 in self.timings:
            if isinstance(self.timings[key1], list):
                if len(self.timings[key1]) > N_samples_to_exclude:
                    avg_timings[key1] = np.mean(self.timings[key1][N_samples_to_exclude:])
            else:
                avg_timings[key1] = {}
                for key2 in self.timings[key1]:
                    if isinstance(self.timings[key1][key2], list):
                        if len(self.timings[key1][key2]) > N_samples_to_exclude:
                            avg_timings[key1][key2] = np.mean(self.timings[key1][key2][N_samples_to_exclude:])
                    else:
                        print("timings... you should not be here")
        return avg_timings

    def WM_eval_deliberateAttentionMechanisms(self, t, T, p_z_s_t, deliberateAttentionMechanisms_to_evaluate, p_z_s_Minus_traces, ltm, dynamicParams, parallel_processing=True):
        p_z_s_t_ = pickle.dumps(p_z_s_t)
        ltm_ = pickle.dumps(ltm)

        evaluations = {}

        if parallel_processing:
            if self.pool is None:
                self.pool = Pool(processes=self.params["n_workers"])
                print("Is going to use " + str(self.params["max_backtracking_evaluations"]+1) + " threads to evaluate deliberateAttentionMechanism!")

            args = []
            for n in range(len(deliberateAttentionMechanisms_to_evaluate)):
                arg = (t, T, p_z_s_t_, deliberateAttentionMechanisms_to_evaluate[n], self.planningImplementation, p_z_s_Minus_traces, ltm_, self.params["N_posterior_samples"], self.pyro_enable_validation, dynamicParams)
                args.append(arg)

            evaluations_ = self.pool.map(WM_eval_deliberateAttentionMechanism, args)

            for evaluation in evaluations_:
                    evaluations[evaluation["deliberateAttentionMechanism"]] = evaluation

        else:
            for n in range(len(deliberateAttentionMechanisms_to_evaluate)): # consider parallel implamentation!
                arg = (t, T, p_z_s_t_, deliberateAttentionMechanisms_to_evaluate[n], self.planningImplementation, p_z_s_Minus_traces, ltm_, self.params["N_posterior_samples"], self.pyro_enable_validation, dynamicParams)
                evaluation = WM_eval_deliberateAttentionMechanism(arg)
                evaluations[evaluation["deliberateAttentionMechanism"]] = evaluation


        for key in evaluations:
            key_ = key.split("_")[0] # if we have multiple of the same entri just numbered e.g. "backtracking_1", "backtracking_2" etc.
            if key_ not in self.timings["infer_action_posterior_est"]:
                self.timings["infer_action_posterior_est"][key_] = []
                self.timings["appraisals_estimation"][key_] = []
            self.timings["infer_action_posterior_est"][key_].append(evaluations[key]["infer_action_posterior_est"])
            self.timings["appraisals_estimation"][key_].append(evaluations[key]["appraisals_estimation"])

        return evaluations

    @abstractmethod
    def WM_reflectiveAttentionMechanism(self, t, T, p_z_s_t, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams):
        # abstract method returning the chosen plan after reflection
        # ...
        # return chosen_plan 
        raise NotImplementedError