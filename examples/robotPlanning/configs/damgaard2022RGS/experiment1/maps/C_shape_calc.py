import numpy as np
import matplotlib.pyplot as plt
import time

pi = 3.14159265359

r = 9/2
center = [7.875, 8.5]
angle_range = [2/3*pi, pi]
direction = -1
N_points = 18
wall_thickness = 1.25

s_sleep = 0.01


if direction < 0:
	step_size = (2*pi-(angle_range[1]-angle_range[0]))/N_points
else:
	step_size = (angle_range[1]-angle_range[0])/N_points
circle_points = []

for i in range(N_points+1):
	angle_ = angle_range[0] + direction*step_size*i
	x = center[0] + (r-wall_thickness/2)*np.cos(angle_)
	y = center[1] + (r-wall_thickness/2)*np.sin(angle_)
	circle_points.append([x, y])

for i in range(N_points+1):
	angle_ = angle_range[1] - direction*step_size*i
	x = center[0] + (r+wall_thickness/2)*np.cos(angle_)
	y = center[1] + (r+wall_thickness/2)*np.sin(angle_)
	circle_points.append([x, y])


circle_points.append(circle_points[0])
point = circle_points[0].copy()
point[0] = 0
circle_points.insert(0,point)
circle_points.append(point)

for i in range(len(circle_points)):
	plt.scatter(circle_points[i][0],circle_points[i][1])
	plt.pause(s_sleep)

print(circle_points)
plt.show()