# robotPlanning
This set of simulations illustrates different usecases of the "Planning" idiom. The simulations utilizes the [HouseExpo](https://github.com/TeaganLi/HouseExpo) dataset and a modified version of their simulator, to generate realistic SLAM output in the form of a map and estimated position.

Before trying any of running any of the simulations remember to unpack files in the HouseExpo simulator as stated in the modules [README](https://github.com/damgaardmr/HouseExpo#getting-started), e.g. by running the command:
```
cd probMind/examples/misc/HouseExpo/HouseExpo
tar -xvzf json.tar.gz
```

## Exploration
<p align="center">
  <img src=Exploration.gif>
</p>

This simulation examplifies pure exploration by utilizing the "Planning" idiom. The goal of the robot is to explore its environment, i.e., determine the state of all cells in a gridmap. The config for this simulation can be found in 
```
probmind/examples/robotPlanning/configs/damgaard22Exploration
```
and the simulation can be started via the command:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python __main__.py -config_file "configs/damgaard22Exploration" -cpu_cores 1
```
For more information we refer you to our publication:
```
@proceedings{damgaard2022AKS,
  editor =       {Malte R. Damgaard and Rasmus Pedersen and Thomas Bak},
  title =        {A Probabilistic Programming Idiom for Active Knowledge Search}, 
  booktitle =    {2022 IEEE World Congress on Computational Intelligence Proceedings},
  publisher =    {IEEE}
  venue =        {Padua, Italy},
  month =        {July},
  year =         {2022},
  isbn =         {978-1-7281-8671-9},
}
```
If you like this work or use in your scientific projects, please use the reference above.

## Goal Search
<p align="center">
  <img src=GoalSearch.gif>
</p>

This simulation examplifies how the "Planning" idiom can be used to find goals in an environment. The simulation builds upon the pure "Exploration" example, by implementing a fallback to exploration whenever the goal is not in sight as illustrated by the GIF above. The config for this simulation can be found in 
```
probmind/examples/robotPlanning/configs/damgaard22GoalSearch
```
and the simulation can be started via the command:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python __main__.py -config_file "configs/damgaard22GoalSearch" -cpu_cores 1
```
### No-Change Impasse
<p align="center">
  <img src=goal_search_without_impasse_detection.gif>
</p>

If the robot only takes actions to get closer to its goal as soon it is visible, the robot can end in a "No-Change Impasse". That is, a situation in which no progress is made as illustrated in the GIF above. This simulations illustrates how the "Planning" idiom can be used to detech such a "No-Change Impasse" and take actions to escape from it as illustrated in the GIF below.

<p align="center">
  <img src=goal_search_with_impasse_detection.gif>
</p>

The config for the simulation abover WITH impasse detection can be found in
```
probmind/examples/robotPlanning/configs/damgaard22GoalSearch_experiment2
```
and the simulation can be started via the command:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python __main__.py -config_file "configs/damgaard22GoalSearch_experiment2" -cpu_cores 1
```
For more information we refer you to our publication:
```
@unpublished{damgaard2022GS,
      title={A Probabilistic Programming Idiom for Goal Search},  
      author={Malte R. Damgaard and Rasmus Pedersen and Thomas Bak},
      year={2022},
      note={Being documented. The Title is tentitative}
}
```
If you like this work or use in your scientific projects, please use the reference above.



## Planning with Multimodal Action Posterior
<p align="center">
  <img src=MultiModalActionPosterior1.gif>
</p>

In environments with highly non-convex representations it is often the case that there exist more than one good plan for the future. This simulation illustrates the use of a multimodal action posterior to represent such ambiguities of the best plan for the future. The simulation builds upon the "Goal Search" example.
The config for this simulation can be found in 
```
probmind/examples/robotPlanning/configs/damgaard22MultiModalActionPosterior
```
and the simulation can be started via the command:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python __main__.py -config_file "configs/damgaard22MultiModalActionPosterior" -cpu_cores 1
```
For more information we refer you to our publication:
```
@unpublished{damgaard2022MAP,
      title={A Probabilistic Programming Idiom for Planning with Multimodal Action Posterior},  
      author={Malte R. Damgaard and Rasmus Pedersen and Thomas Bak},
      year={2022},
      note={Being documented. The Title is tentitative}
}
```
If you like this work or use in your scientific projects, please use the reference above.


## Cluster Computing
The "__main__.py" script has the three addtional args that can be used to divide the simulation task onto a cluster of computers with different hardware capabilities. The "-cpu_cores" arg determines the number of independent simulations being started on a single computer. The "-thread_start_ID" arg offsets the ID's of the simulations being runned at a specific computer. The "-total_cluster_threads" arg is used to divide the computational burden evenly. These args implement very basic logic and assumes that processors of each computer is roughly equal fast. As an example, if you want to run a simulation on a cluster with 3 individual computers with 1 of them having 8 cpu cores and the others having 4 cpu cores, then you can divide the computational burden of a simulation by running the following commands on each pc:
```
$ python __main__.py -cpu_cores 8 -thread_start_ID 0 -total_cluster_threads 16  # on the pc with 8 cores
$ python __main__.py -cpu_cores 4 -thread_start_ID 8 -total_cluster_threads 16  # on a pc with 4 cores
$ python __main__.py -cpu_cores 4 -thread_start_ID 12 -total_cluster_threads 16  # on a pc with 4 cores
```


## TODO
The following is a list of things that could potentially be improved:
- [ ] implement rotation and FOV in lidar model
- [ ] Change state to a dict - e.g. z_s_tau["position"]
- [X] consider noise on state in the simulator


## Collaboration/Contributions/Questions
If you would like to collaborate on this project or anything related, have any questions, or make contributions to the project, feel free to reach out to [Malte R. Damgaard](https://vbn.aau.dk/da/persons/134312) prefeerable via [email](mailto:mrd@es.aau.dk?subject=[GitHub]%20The%20ProbMind%20Project)
