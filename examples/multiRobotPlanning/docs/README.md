# multiRobotPlanning
<p align="center">
  <img src=example.gif>
</p>

This simulation illustrate the use of the "Planning" idiom for multi robot navigation. The motivation behind this simulation is two folds:
- It illustrates how SVI and message-passing can be combined to de-centralize the computational burden of complex problems
- It illustrates a simplistic implementation of Theory of Mind

The goal of each robot is to navigate between two alternating goal zones without colliding with the other robots. To avoid collsions the robots sends messages with their current plan to the other robots wihtin a certain distance. The algorithmic details are descriped in our publication:
```
@unpublished{damgaard2022SVIFDPR,
      title={Study of Variational Inference for Flexible Distributed Probabilistic Robotics}, 
      author={Malte R. Damgaard and Rasmus Pedersen and Thomas Bak},
      year={2022},
      note={Submitted}
}
```
If you like this work or use it in your scientific projects, please use the reference above. The config for the simulation detailed in the above publication can be found in 
```
probmind/examples/multiRobotPlanning/configs/damgaard2022SVIFDPR
```
and the simulation can be started via the command:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python __main__.py -config_file "configs/damgaard2022SVIFDPR/configs.yaml"
```
When running the simulations be aware of the number af messages sent between the robots, which will be printed to the terminal. On older hardware the number of messages sent pr. timestep may fall below 3-4, and you will have to adjust til realtime factor accordingly.

## ROS2 Implementation
The same algorithm have been implemented in ROS2 and tested on turlebot3. The code is availeable at ![ROS2 Implementation](https://github.com/damgaardmr/VI_Nav)
