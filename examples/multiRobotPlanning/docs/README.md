# multiRobotPlanning
<p align="center">
  <img src=example.gif>
</p>

These simulations illustrates the use of the "Planning" idiom for multi robot navigation. The motivation behind the simulations is two folds:
- They illustrate how SVI and message-passing can be combined to de-centralize the computational burden of complex problems
- They illustrate a simplistic implementation of Theory of Mind

The goal of each robot is to navigate between two alternating goal zones without colliding with the other robots. To avoid collsions the robots sends messages with their current plan to the other robots wihtin a certain distance. The algorithmic details are descriped in our publication:
```
@Article{robotics11020038,
  AUTHOR = {Damgaard, Malte Rørmose and Pedersen, Rasmus and Bak, Thomas},
  TITLE = {Study of Variational Inference for Flexible Distributed Probabilistic Robotics},
  JOURNAL = {Robotics},
  VOLUME = {11},
  YEAR = {2022},
  NUMBER = {2},
  ARTICLE-NUMBER = {38},
  URL = {https://www.mdpi.com/2218-6581/11/2/38},
  ISSN = {2218-6581},
  ABSTRACT = {By combining stochastic variational inference with message passing algorithms, we show how to solve the highly complex problem of   navigation and avoidance in distributed multi-robot systems in a computationally tractable manner, allowing online implementation. Subsequently, the proposed variational method lends itself to more flexible solutions than prior methodologies. Furthermore, the derived method is verified both through simulations with multiple mobile robots and a real world experiment with two mobile robots. In both cases, the robots share the operating space and need to cross each other&rsquo;s paths multiple times without colliding.},
  DOI = {10.3390/robotics11020038}
}
```
If you like this work or use it in your scientific projects, please use the reference above. The config for the simulations detailed in the above publication can be found in 
```
probmind/examples/multiRobotPlanning/configs/damgaard2022SVIFDPR/<Simulation_Scenario>/
```
The simulation for the "Antipodal Circle Goal" scenario can be started via the commands:
```
$ cd probmind/examples/robotPlanning
$ conda activate probMind_env_cross_platform
$ python sim.py -config_file "configs/damgaard2022SVIFDPR/Antipodal_Circle/configs.yaml"
```
Similarely, the all the simulations of 2, 4, 8, 16, and 32 robots for the "Antipodal Circle Swapping" scenario can be started via the command:
```
$ python damgaard2022SVIFDPR_Antipodal_Circle_Swapping.py
```
When running the simulations be aware of the number af messages sent between the robots, which will be printed to the terminal. On older hardware the number of messages sent pr. timestep may fall below 3-4, and you will have to adjust til realtime factor accordingly. The data and videos of the simulations in the publications is contained in the folder:
```
probmind/examples/multiRobotPlanning/DATA/damgaard2022SVIFDPR/<Simulation_Scenario>/
```

## ROS2 Implementation
The same algorithm have been implemented in ROS2 and tested on turlebot3. The code is availeable at ![ROS2 Implementation](https://github.com/damgaardmr/VI_Nav)
