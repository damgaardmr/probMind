import torch
from torch.distributions import constraints
import pyro
import pyro.distributions as dist
from pyro import poutine
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time

from lidarmodel import p_z_Lidar_prior, p_z_Lidar_posterior, p_z_Map_prior, lidar_generate_labels
from barrierfunctions import logistic
import dill as pickle

from pathlib import Path
from sys import path
import os
from os.path import dirname as dir
path.append(dir(Path(path[0].replace("robotPlanning", ""))))
__package__ = "probmind"

from cognition.planning import Planning
from cognition.ltm import LTM
from cognition.appraisals import Appraisals
from cognition.reflectiveAttentionMechanism import ReflectiveAttentionMechanism, WM_eval_deliberateAttentionMechanism
from cognition.misc import KL_point_estimate
from cognition.stateTree import StateTreeBranch, flattenList
from cognition.deliberateAttentionMechanism import ConstraintAvoidance, StateReach, StateReachWithProgress, ExploreWithoutConstraint, StateReachWithExplore, Explore, ExploreWithProgress, Progress


class ReflectiveAttentionMechanismImplementation(ReflectiveAttentionMechanism):
    def __init__(self, appraisalsImplementation, PlanningImplementation, params):
        n_workers = max([params["max_backtracking_evaluations"]+4, 5])
        super().__init__(appraisalsImplementation, PlanningImplementation, params, n_workers=n_workers)

        if "Disable_Impasse" in params:
            self.disable_Impasse = params["Disable_Impasse"]
        else:
            self.disable_Impasse = False

        if "always_go_to_goal_when_visable" in params:
            self.always_go_to_goal_when_visable = params["always_go_to_goal_when_visable"]
        else:
            self.always_go_to_goal_when_visable = False

        self.chosen_plan = None
        self.ancestorsStateIndexes = [0]
        self.rootStateBranch = StateTreeBranch(stateIndexes=[])
        self.currentStateBranch = self.rootStateBranch
        self.rootStateHaveBeenVisited = False
        self.branchingStateIndexes = []


    deliberateAttentionMechanisms = {"StateReach": StateReach,
                                     "StateReachWithProgress": StateReachWithProgress,
                                     "StateReachWithExplore":StateReachWithExplore,
                                     "Explore": Explore,
                                     "ExploreWithProgress": ExploreWithProgress,
                                     "ExploreWithoutConstraint": ExploreWithoutConstraint,
                                     "Progress": Progress,
                                     "ConstraintAvoidance": ConstraintAvoidance} # Progress might not be necessary

    def reset(self):
        super().reset()
        self.ancestorsStateIndexes = [0]
        self.rootStateBranch = StateTreeBranch(stateIndexes=[])
        self.currentStateBranch = self.rootStateBranch
        self.chosen_plan = None
        self.rootStateHaveBeenVisited = False
        self.branchingStateIndexes = []


    def WM_reflectiveAttentionMechanism(self, t, T, p_z_s_t, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams):
        self.branchingStateIndexes = flattenList(self.currentStateBranch.getBranchingPoints().copy())

        deliberateAttentionMechanisms_evaluated = {}
        if self.chosen_plan is None:
            if p_z_g is None:
                self.chosen_plan,_ = self.WM_reflectiveAttentionMechanism_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams, deliberateAttentionMechanism="Explore")
            else:
                self.chosen_plan,_ = self.WM_reflectiveAttentionMechanism_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams)
            return self.chosen_plan
        else:
            self.chosen_plan,_ = self.WM_reflectiveAttentionMechanism_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams, deliberateAttentionMechanism=self.chosen_plan["deliberateAttentionMechanism"])
            return self.chosen_plan

    def WM_reflectiveAttentionMechanism_(self, t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams, deliberateAttentionMechanism="StateReach", recursions=0):
        if self.disable_Impasse:
            deliberateAttentionMechanism = "StateReach"

        if not deliberateAttentionMechanism.startswith("Backtracking"):
            if recursions==0:
                if self.always_go_to_goal_when_visable and p_z_g is not None:
                    deliberateAttentionMechanism = "StateReach"

        deliberateAttentionMechanism = deliberateAttentionMechanism.replace("_constraintAvoidance","")
        tic1 = time.time()
        deliberateAttentionMechanisms_to_evaluate = self.WM_deliberate_attention_proposal_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, deliberateAttentionMechanism, dynamicParams)
        tic2 = time.time()
        deliberateAttentionMechanisms_evaluated.update(self.WM_eval_deliberateAttentionMechanisms(t, T, p_z_s_t, deliberateAttentionMechanisms_to_evaluate, p_z_s_Minus_traces, ltm, dynamicParams))
        tic3 = time.time()
        chosen_plan, recursions_ = self.WM_affective_responses_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, deliberateAttentionMechanism, recursions, dynamicParams)
        toc = time.time()

        # save timings...
        if deliberateAttentionMechanism.startswith("StateReach"):
            timings_key = "StateReach"
        elif deliberateAttentionMechanism.startswith("Explore"):
            timings_key = "Explore"
        elif deliberateAttentionMechanism.startswith("Backtracking"):
            timings_key = "Backtracking"
        else:
            timings_key = deliberateAttentionMechanism

        if timings_key not in self.timings:
            self.timings[timings_key] = {}
            self.timings[timings_key]["deliberate_attention_proposal"] = []
            self.timings[timings_key]["deliberate_Attention_evaluation"] = []
            self.timings[timings_key]["affective_responses"] = []

        self.timings[timings_key]["deliberate_attention_proposal"].append(tic2-tic1)
        self.timings[timings_key]["deliberate_Attention_evaluation"].append(tic3-tic2)
        if recursions_ == recursions:
            self.timings[timings_key]["affective_responses"].append(toc-tic3)
        else:
            pass # how to properly count time when recursions has occured?

        return chosen_plan, recursions_


    def WM_deliberate_attention_proposal_(self, t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, deliberateAttentionMechanism, dynamicParams):

        if deliberateAttentionMechanism == "StateReach" or  deliberateAttentionMechanism == "StateReachWithProgress" or deliberateAttentionMechanism == "StateReachWithExplore":
            if p_z_g is not None:
                # calculate kl-divergence
                with poutine.block():
                    p_z_g_trace = poutine.trace(p_z_g).get_trace()
                    desirability_KL_baseline = KL_point_estimate(t, poutine.trace(p_z_s_t).get_trace(), p_z_g_trace).detach()       # baseline relative to current state
            else:
                desirability_KL_baseline = None

            deliberateAttentionMechanisms_to_evaluate = []
            deliberateAttentionMechanisms_needed = ["ConstraintAvoidance", "StateReach", "StateReachWithProgress", "StateReachWithExplore"]
            deliberateAttentionMechanisms_to_evaluate_ = list(set(deliberateAttentionMechanisms_needed) - set(deliberateAttentionMechanisms_evaluated.keys()))
            for deliberateAttentionMechanism_key in deliberateAttentionMechanisms_to_evaluate_:
                deliberateAttentionMechanism_ = self.deliberateAttentionMechanisms[deliberateAttentionMechanism_key](self.appraisalsImplementation, ltm, self.params, p_z_g=p_z_g, desirability_KL_baseline=desirability_KL_baseline)
                deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanism_) # here we include the goal only because it is needed in the evaluations of all appraisals




        elif deliberateAttentionMechanism == "Explore" or deliberateAttentionMechanism == "ExploreWithProgress":
            deliberateAttentionMechanisms_to_evaluate = []
            if p_z_g is None:
                deliberateAttentionMechanisms_needed = ["ConstraintAvoidance", "Explore", "ExploreWithProgress"]
                desirability_KL_baseline = None
            else:
                deliberateAttentionMechanisms_needed = ["ConstraintAvoidance", "StateReach", "Explore", "ExploreWithProgress"]
                # calculate kl-divergence
                with poutine.block():
                    p_z_g_trace = poutine.trace(p_z_g).get_trace()
                    desirability_KL_baseline = KL_point_estimate(t, poutine.trace(p_z_s_t).get_trace(), p_z_g_trace).detach()       # baseline relative to current state
            deliberateAttentionMechanisms_to_evaluate_ = list(set(deliberateAttentionMechanisms_needed) - set(deliberateAttentionMechanisms_evaluated.keys()))
            for deliberateAttentionMechanism_key in deliberateAttentionMechanisms_to_evaluate_:
                deliberateAttentionMechanism_ = self.deliberateAttentionMechanisms[deliberateAttentionMechanism_key](self.appraisalsImplementation, ltm, self.params, p_z_g=p_z_g, desirability_KL_baseline=desirability_KL_baseline)
                deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanism_) # here we include the goal only because it is needed in the evaluations of all appraisals


        elif deliberateAttentionMechanism == "Backtracking":
            deliberateAttentionMechanisms_to_evaluate = []
            deliberateAttentionMechanisms_needed = ["ConstraintAvoidance", "Explore", "ExploreWithProgress"] #, "ExploreWithoutConstraint", "Progress"]  # backtracking states are treated seperately
            deliberateAttentionMechanisms_to_evaluate_ = list(set(deliberateAttentionMechanisms_needed) - set(deliberateAttentionMechanisms_evaluated.keys()))
            for deliberateAttentionMechanism_key in deliberateAttentionMechanisms_to_evaluate_:
                deliberateAttentionMechanism_ = self.deliberateAttentionMechanisms[deliberateAttentionMechanism_key](self.appraisalsImplementation, ltm, self.params, p_z_g=None, desirability_KL_baseline=None)
                deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanism_) # here we include the goal only because it is needed in the evaluations of all appraisals

            #self.state_tree_handling4()
            self.n_backtracking_actually_evaluated = -1
            for i in range(self.params["max_backtracking_evaluations"]):
                p_z_g_backtracking, backtracking_KL_baseline = self.get_backtracking_goal(t, p_z_s_Minus, p_z_s_Minus_traces, delta_t=i)
                if p_z_g_backtracking is not None:
                    self.n_backtracking_actually_evaluated = self.n_backtracking_actually_evaluated + 1
                    deliberateAttentionMechanisms_ = StateReach(self.appraisalsImplementation, ltm, self.params, dynamicParams, p_z_g=p_z_g_backtracking, desirability_KL_baseline=backtracking_KL_baseline.detach(), name="Backtracking_"+str(i))
                    deliberateAttentionMechanisms_to_evaluate.append(deliberateAttentionMechanisms_)

        else:
            print("You should not be here! missing a case for current deliberateAttentionMechanism!")
      
        return deliberateAttentionMechanisms_to_evaluate



    def WM_affective_responses_(self, t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, deliberateAttentionMechanism, recursions, dynamicParams):

        if deliberateAttentionMechanism == "StateReach" or  deliberateAttentionMechanism == "StateReachWithProgress" or deliberateAttentionMechanism == "StateReachWithExplore":
            self.add_state_to_stateTree(t)
            P_z_p_est = {}
            for key in ["StateReach", "StateReachWithProgress", "StateReachWithExplore"]:
                P_z_p_est[key] = deliberateAttentionMechanisms_evaluated[key]["P_alpha_T_expectations"]["progress"]

            no_change = len([key for key, P_z_p_est_ in P_z_p_est.items() if P_z_p_est_ > self.params["P_z_p_lim"]]) == 0

            if no_change and not self.disable_Impasse: # No_Change_Impasse
                return self.WM_reflectiveAttentionMechanism_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams, deliberateAttentionMechanism="Explore", recursions=recursions+1)

            else:
                P_z_d_est = {}
                keys = P_z_p_est.keys()

                for key in keys:
                    P_z_d_est[key] = deliberateAttentionMechanisms_evaluated[key]["P_alpha_T_expectations"]["desirability"]

                P_z_d_est = {k: v for k, v in sorted(P_z_d_est.items(), key=lambda item: item[1], reverse=True)} # sort after highest desirbility first
                P_z_C_est2 = deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["P_alpha_T_expectations"]["constraints accumulated"]

                for key in P_z_d_est:
                    if P_z_d_est[key] >= P_z_d_est["StateReach"]: # only choose plans that brings us closer to the goal than the method going straight towards the goal
                        chosen_plan = deliberateAttentionMechanisms_evaluated[key]
                        
                        P_z_C_est1 = chosen_plan["P_alpha_T_expectations"]["constraints accumulated"]
                        if (P_z_C_est1 >= self.params["P_z_C_lim"] or # return most desirable plan satiesfying constraints!
                            P_z_C_est1 >= P_z_C_est2): # if the statereach is chosen_plan than the constraint avoidance plan.)
                            return chosen_plan, recursions

                # if no plan satiesfies constraints return ConstraintAvoidance-plan
                deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["deliberateAttentionMechanism"] = chosen_plan["deliberateAttentionMechanism"]+"_constraintAvoidance"# to enter the same reflective mechanism next iteration
                return deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"], recursions




        elif deliberateAttentionMechanism == "Explore" or deliberateAttentionMechanism == "ExploreWithProgress":
            self.add_state_to_stateTree(t)
            P_z_p_est = {}
            P_z_i_est = {}
            for key in ["Explore", "ExploreWithProgress"]:#, "ExploreWithoutConstraint", "Progress"]:
                P_z_p_est[key] = deliberateAttentionMechanisms_evaluated[key]["P_alpha_T_expectations"]["progress"]
                P_z_i_est[key] = deliberateAttentionMechanisms_evaluated[key]["P_alpha_T_expectations"]["information gain"]

            no_change = len([key for key, P_z_p_est_ in P_z_p_est.items() if P_z_p_est_ > self.params["P_z_p_lim"]]) == 0

            P_z_i_est = {k: v for k, v in sorted(P_z_i_est.items(), key=lambda item: item[1], reverse=True)} # sort after highest information gain first
            max_key = list(P_z_i_est.keys())[0]

            if P_z_i_est[max_key] <= self.params["P_z_i_lim"] or no_change: # No information to obtain or no change impasse
                self.set_path_to_backtrack(t)
                return self.WM_reflectiveAttentionMechanism_(t, T, p_z_s_t, deliberateAttentionMechanisms_evaluated, p_z_s_Minus, p_z_s_Minus_traces, ltm, p_z_g, dynamicParams, deliberateAttentionMechanism="Backtracking", recursions=recursions+1)

            else:
                if p_z_g is not None:
                    P_z_p_est2 = deliberateAttentionMechanisms_evaluated["StateReach"]["P_alpha_T_expectations"]["progress"]
                    P_z_i_est2 = deliberateAttentionMechanisms_evaluated["StateReach"]["P_alpha_T_expectations"]["information gain"]
                P_z_C_est2 = deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["P_alpha_T_expectations"]["constraints accumulated"]
                for key in P_z_i_est:
                    chosen_plan = deliberateAttentionMechanisms_evaluated[key]
                    if p_z_g is not None:
                        if P_z_i_est2 >= P_z_i_est[key]*(1-self.params["P_z_i_delta"]) and P_z_p_est2 > self.params["P_z_p_lim"]:
                            chosen_plan = deliberateAttentionMechanisms_evaluated["StateReach"]

                    P_z_C_est1 = chosen_plan["P_alpha_T_expectations"]["constraints accumulated"]
                    if (P_z_C_est1 >= self.params["P_z_C_lim"] or # possible constraint violation!
                        P_z_C_est1 >= P_z_C_est2): # if the chosen_plan is better than the constraint avoidance plan.
                        return chosen_plan, recursions

                #P_z_C_est1 = chosen_plan["P_alpha_T_expectations"]["constraints accumulated"]
                #if (P_z_C_est1 >= self.params["P_z_C_lim"] or 
                #    P_z_C_est1 >= P_z_C_est2): # if the chosen_plan is better than the constraint avoidance plan.
                #    return chosen_plan, recursions
                #else:

                # if no plan satiesfies constraints return ConstraintAvoidance-plan
                deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["deliberateAttentionMechanism"] = chosen_plan["deliberateAttentionMechanism"]+"_constraintAvoidance"# to enter the same reflective mechanism next iteration
                return deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"], recursions




        elif deliberateAttentionMechanism == "Backtracking":
            P_z_i_est = {}
            for key in ["Explore", "ExploreWithProgress"]: #, "ExploreWithoutConstraint", "Progress"]:
                P_z_i_est[key] = deliberateAttentionMechanisms_evaluated[key]["P_alpha_T_expectations"]["information gain"]

            max_key = max(P_z_i_est, key=P_z_i_est.get)

            P_z_p_est1 = deliberateAttentionMechanisms_evaluated[max_key]["P_alpha_T_expectations"]["progress"]
            P_z_C_est2 = deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["P_alpha_T_expectations"]["constraints accumulated"]
            if (P_z_i_est[max_key] > self.params["P_z_i_lim"] and P_z_p_est1 > self.params["P_z_p_lim"]): # or self.n_backtracking_actually_evaluated == -1:
                self.create_new_state_branch(t)
                chosen_plan = deliberateAttentionMechanisms_evaluated[max_key]
                P_z_C_est1 = chosen_plan["P_alpha_T_expectations"]["constraints accumulated"]
            else:
                for i in range(self.n_backtracking_actually_evaluated, -1, -1):
                    chosen_plan = deliberateAttentionMechanisms_evaluated["Backtracking_"+str(i)]
                    P_z_d_est = chosen_plan["P_alpha_T_expectations"]["desirability"]
                    P_z_C_est1 = chosen_plan["P_alpha_T_expectations"]["constraints accumulated"]
                    if (P_z_d_est > self.params["P_min_backtracking"] and
                       (P_z_C_est1 >= self.params["P_z_C_lim"] or           # possible constraint violation!
                        P_z_C_est1 >= P_z_C_est2)):                         # if the chosen_plan is better than the constraint avoidance plan.):
                        self.should_state_be_removed_from_path(P_z_d_est, delta_t=i)
                        self.has_end_of_path_been_reached()
                        #break
                        chosen_plan["deliberateAttentionMechanism"] = "Backtracking"
                        return chosen_plan, recursions

                chosen_plan["deliberateAttentionMechanism"] = "Backtracking"

            if (P_z_C_est1 >= self.params["P_z_C_lim"] or # possible constraint violation!
                P_z_C_est1 >= P_z_C_est2): # if the chosen_plan is than the constraint avoidance plan.)
                return chosen_plan, recursions
            else:
                deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"]["deliberateAttentionMechanism"] = chosen_plan["deliberateAttentionMechanism"]+"_constraintAvoidance"# to enter the same reflective mechanism next iteration
                return deliberateAttentionMechanisms_evaluated["ConstraintAvoidance"], recursions

        else:
            print("You should not be here! missing a case for current deliberateAttentionMechanism!")








    def set_path_to_backtrack(self, t):
        if self.rootStateHaveBeenVisited:
            self.ancestorsStateIndexes = self.currentStateBranch.getRandomPath().copy()
        else: # the first time backtracking we should go all the way bact to the start to make sure that the initial state have been visited
            self.ancestorsStateIndexes = self.currentStateBranch.getAncestorsStateIndexes().copy()

    def create_new_state_branch(self, t):
        if self.currentStateBranch.stateIndexes: # list is not empty
            self.currentStateBranch.splitBranch(self.ancestorsStateIndexes[0])
            self.currentStateBranch = self.currentStateBranch.children[-1]

    def add_state_to_stateTree(self, t):
        if t not in self.currentStateBranch.stateIndexes:
            self.currentStateBranch.addStateIndex(t)

    def has_end_of_path_been_reached(self):
        if len(self.ancestorsStateIndexes) == 1: # we reached end of backtracking try
            if self.ancestorsStateIndexes[0] == 0:
                print("reached end of backtracking, trying a random branch instead...")
                self.rootStateHaveBeenVisited = True
            else:
                print("reached end of path, trying a another random branch instead...")
            self.ancestorsStateIndexes = self.currentStateBranch.getRandomPath().copy()

    def should_state_be_removed_from_path(self, P_z_d, delta_t=1):
        #if P_z_d > self.params["P_min_backtracking"]: <-- currently redundant
        for tau in range(delta_t):
            if len(self.ancestorsStateIndexes) > 1:
                self.ancestorsStateIndexes.pop(0)
                if self.ancestorsStateIndexes[0] not in self.currentStateBranch.stateIndexes:
                    oldBranch = self.currentStateBranch.stateIndexes
                    changedBranch = False
                    if self.currentStateBranch.parent is not None:
                        if self.ancestorsStateIndexes[0] in self.currentStateBranch.parent.stateIndexes:
                            changedBranch = True
                            self.currentStateBranch = self.currentStateBranch.parent
                    
                    if self.currentStateBranch.children is not None:
                        for child in self.currentStateBranch.children:
                            if self.ancestorsStateIndexes[0] in child.stateIndexes:
                                changedBranch = True
                                self.currentStateBranch = child
                                break
                    newBranch = self.currentStateBranch.stateIndexes
                    if not changedBranch:
                        print("You should not have been here...542")

    def get_backtracking_goal(self, t, p_z_s_Minus, p_z_s_Minus_traces, delta_t=0):
        if delta_t < len(self.ancestorsStateIndexes):
            p_z_g_backtracking = p_z_s_Minus[self.ancestorsStateIndexes[delta_t]]
            with poutine.block():
                p_z_g_trace = poutine.trace(p_z_g_backtracking).get_trace()
                backtracking_KL_baseline = KL_point_estimate(t, p_z_s_Minus_traces[t], p_z_g_trace)

            return p_z_g_backtracking, backtracking_KL_baseline
        else:
            return None, None
