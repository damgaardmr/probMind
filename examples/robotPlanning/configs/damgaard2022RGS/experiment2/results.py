import os
import pickle
import lzma
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tikzplotlib
# latex formatting strings:
# https://matplotlib.org/stable/tutorials/text/usetex.html
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
# remember that special charathers in strings in figures have to have prober escaping: i.e. use "\%" instead of "%"
#plt.rcParams['text.usetex'] = True 


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


def max_area_explored(max_step_length, lidar_radius, N_timesteps):
	C1 = {"x": 0, "y": 0, "r": lidar_radius}
	C2 = C1.copy()
	C2["x"] = max_step_length

	def circle_area(C):
		return np.pi*C["r"]*C["r"]


	def circle_intersection_area(C1, C2):
		# inspired by https://www.xarg.org/2016/07/calculate-the-intersection-area-of-two-circles/
	    d = np.hypot(C2["x"] - C1["x"], C2["y"] - C1["y"])
	    if (d < C1["r"] + C2["r"]):
	        a = C1["r"] * C1["r"]
	        b = C2["r"] * C2["r"]

	        x = (a - b + d * d) / (2 * d)
	        z = x * x
	        y = np.sqrt(a - z)

	        if (d <= np.abs(C2["r"] - C1["r"])):
	            return PI * np.min(a, b)
	        else:
	        	return a * np.arcsin(y / C1["r"]) + b * np.arcsin(y / C2["r"]) - y * (x + np.sqrt(z + b - a))
	    else:
	    	return 0

	LidarArea = circle_area(C1)
	intersection = circle_intersection_area(C1, C2)

	LidarArea_ = LidarArea - intersection

	max_area_explored = LidarArea
	for n in range(1, N_timesteps):
		max_area_explored = max_area_explored + LidarArea_

	return max_area_explored



# ############################## setup ##############################
colors = {
    "yellow": "#FBBC05",
    "green": "#34A853",
    "red": "#EA4335",
    "blue": "#4285F4",
    "purple": "#410093",
    "pink": "#FF33FF",
    "oliveGreen": "#AFB83B",
    "black": "black",
}

#DATA_FOLDER = "../../../DATA"
DATA_FOLDER = "DATA"
JSONfileDir = "../../../../misc/HouseExpo/HouseExpo/json"
tikzFolder = "tikzPlots"

experiments = {"damgaard2022AKS": {"dataFolder": "damgaard2022AKS", "color": colors["blue"]},
		       "damgaard2022RGS": {"dataFolder": "damgaard2022RGS", "color": colors["yellow"]},
		       "damgaard2022RGS_small": {"dataFolder": "damgaard2022RGS_small", "color": colors["green"]}}


Time_steps_max = 200
N = 100 # window size 
linewidth = 3 # for plotting
alpha = 0.2

limitNumberOfSims = False
numberOfSims = 300

# ############################## load data ##############################
resultsDATAfile = DATA_FOLDER + "/" + "resultsDATA" + ".xz"
if os.path.exists(resultsDATAfile):
	f = lzma.open(resultsDATAfile, 'rb')
	try:
		data = pickle.load(f)
	finally:  # to close file in case of exception
		f.close()
else:
	data = {}

if True:
	for experiment in experiments:
		print("Loading data for experiment: " + str(experiment))
		DATA_FOLDER_ = DATA_FOLDER + "/" + experiments[experiment]["dataFolder"]

		files_ = list_files(DATA_FOLDER_)
		if experiment not in data:
			data[experiment] = {}

		counter = 0
		n_files = len(files_)
		for file in files_:
			counter = counter + 1
			if file.endswith(".xz"):
				map_ID = file.replace(DATA_FOLDER_+"/","").partition("/")[0]
				if map_ID not in data[experiment]:
					f = lzma.open(file, 'rb')
					try:
						data_ = pickle.load(f)

						Map_ID = data_["Map_ID"]
						data[experiment][Map_ID] = {}

						timesteps = data_["metrics"]["time_steps_used"]
						data[experiment][Map_ID]["timesteps"] = timesteps
						if timesteps == Time_steps_max:
							data[experiment][Map_ID]["finished"] = False
						else:
							data[experiment][Map_ID]["finished"] = True

						data[experiment][Map_ID]["meter2pixel"] = data_["meter2pixel"]
						data[experiment][Map_ID]["total_percentage_explored"] = data_["metrics"]["total_percentage_explored"]
						data[experiment][Map_ID]["true_map_area_in_pixels"] = data_["metrics"]["true_map_area_in_pixels"]
						data[experiment][Map_ID]["total_percentage_explored"] = data_["metrics"]["total_percentage_explored"]

						jsonFile = JSONfileDir + "/" + Map_ID + ".json"
						with open(jsonFile, 'rb') as f_json:
							json_data_map_i = json.load(f_json)
							data[experiment][Map_ID]["n_rooms"] = json_data_map_i['room_num']

						# using all collisions
						# data[experiment][Map_ID]["number_of_collisions"] = data_["metrics"]["number_of_collisions"]

						# using distance "if consequtive equal"
						if data_["collisions"]: # not empty
							min_dist = 1/data_["meter2pixel"]  # meter
							collisions = [data_["collisions"][0]]
							for k in range(1, len(data_["collisions"])):
								a = data_["collisions"][k - 1]
								b = data_["collisions"][k]
								dist = np.linalg.norm(a - b)
								if dist > min_dist:  # if smaller than min_dist we do not count it
									collisions.append(b)
							data[experiment][Map_ID]["number_of_collisions"] = len(collisions)
						else:
							data[experiment][Map_ID]["number_of_collisions"] = 0
					except:
						print("An exception occurred for map_ID: " + str(map_ID))

			if counter % int(n_files/10) == 0:
				percentageLoaded = counter/len(files_)*100
				print("Loaded {percentageLoaded:.2f} %".format(percentageLoaded=percentageLoaded))

	with lzma.open(resultsDATAfile, "wb") as f:
	    pickle.dump(data, f)


# ############################## process data ##############################
# make a list of maps available in both experiments set
#for i in range(len(experiments)):
#	print(len(list(data[list(experiments.keys())[i]].keys())))

map_IDs = set(list(data[list(experiments.keys())[0]].keys()))
for experiment in list(experiments.keys())[1:]:
	map_IDs_ = list(data[experiment].keys())
	map_IDs = map_IDs.intersection(map_IDs_)

map_IDs = list(map_IDs)

truearea = pd.DataFrame(columns = ["pixels","m2","n_rooms"], index = map_IDs)
truearea.index.name = "mapIDs"
for map_ID in map_IDs:
	truearea["pixels"][map_ID] = data[list(experiments.keys())[0]][map_ID]["true_map_area_in_pixels"]
	truearea["m2"][map_ID] = data[list(experiments.keys())[0]][map_ID]["true_map_area_in_pixels"]/(data[list(experiments.keys())[0]][map_ID]["meter2pixel"]**2)
	truearea["n_rooms"][map_ID] = data[list(experiments.keys())[0]][map_ID]["n_rooms"]

truearea = truearea.sort_values(by=["pixels"])


max_step_length = 0.3282069027458667 # found using the "play_trajectory.py" script
lidar_radius = 2.0
N_timesteps = 100
max_are_explored_with_backtracting = max_area_explored(max_step_length, lidar_radius, N_timesteps)
tmp = truearea[truearea["m2"] > max_are_explored_with_backtracting]
try:
	max_are_explored_with_backtracting_idx = truearea[truearea["m2"] > max_are_explored_with_backtracting].index[0]
	max_are_explored_with_backtracting_idx_number = list(truearea.index).index(max_are_explored_with_backtracting_idx)
except:
	max_are_explored_with_backtracting_idx_number = np.nan

N_timesteps = 200
max_are_explored_without_backtracting = max_area_explored(max_step_length, lidar_radius, N_timesteps)
#max_are_explored_without_backtracting_idx = truearea[truearea["m2"] > max_are_explored_without_backtracting].index[0]
#max_are_explored_without_backtracting_idx_number = list(truearea.index).index(max_are_explored_without_backtracting_idx)


if limitNumberOfSims:
	map_IDs = list(truearea.index)[0:numberOfSims]
	map_IDs = list(truearea.index)[0:max_are_explored_with_backtracting_idx_number]
	if len(map_IDs) < max_are_explored_with_backtracting_idx_number:
		max_are_explored_with_backtracting_idx_number = np.nan
	truearea = truearea.loc[map_IDs]
print("N maps: " + str(len(map_IDs)))


keys_ = list(experiments.keys())
for experiment in list(experiments.keys()):
	keys_.append(experiment+"_smoothed")

timesteps = pd.DataFrame(columns = list(experiments.keys()), index = list(truearea.index))
timesteps.index.name = "mapIDs"
percentage_explored = pd.DataFrame(columns = keys_, index = list(truearea.index))
percentage_explored.index.name = "mapIDs"
area_explored = pd.DataFrame(columns = keys_, index = list(truearea.index))
area_explored.index.name = "mapIDs"
collisions = pd.DataFrame(columns = list(experiments.keys()), index = list(truearea.index))
collisions.index.name = "mapIDs"
finished = pd.DataFrame(columns = list(experiments.keys()), index = list(truearea.index))
finished.index.name = "mapIDs"

for experiment in list(experiments.keys()):
	for map_ID in map_IDs:
		timesteps[experiment][map_ID] = data[experiment][map_ID]["timesteps"]
		percentage_explored[experiment][map_ID] = data[experiment][map_ID]["total_percentage_explored"]
		area_explored[experiment][map_ID] = data[experiment][map_ID]["total_percentage_explored"]*truearea["m2"][map_ID]
		collisions[experiment][map_ID] = data[experiment][map_ID]["number_of_collisions"]
		finished[experiment][map_ID] = data[experiment][map_ID]["finished"]

	smooted_percentage = np.convolve(percentage_explored[experiment], np.ones(N) / N, mode='valid')
	percentage_explored[experiment+"_smoothed"][list(truearea.index)[N-1:]] = smooted_percentage
	
	smooted_area = np.convolve(area_explored[experiment], np.ones(N) / N, mode='valid')
	#area_explored[experiment+"_smoothed"][list(truearea.index)[N-1:]] = smooted_area
	idx1 = int(N/2)
	idx2 = idx1 + len(smooted_area)
	area_explored[experiment+"_smoothed"][list(truearea.index)[idx1:idx2]] = smooted_area


map_IDs_finished_by_all = set(list(finished[finished[list(experiments)[0]]==True].index))
#print(len(map_IDs_finished_by_all))
for i in range(1,len(list(experiments))):
	map_IDs_ = set(list(finished[finished[list(experiments)[i]]==True].index))
	map_IDs_finished_by_all = map_IDs_finished_by_all.intersection(map_IDs_)
	#print(len(map_IDs_finished_by_all))


cols_ = []
for experiment in experiments:
	cols_.append((experiment,"collision"))
	cols_.append((experiment,"n_collision"))
	cols_.append((experiment,"notFinished"))
	cols_.append((experiment,"Both"))
cols = pd.MultiIndex.from_tuples(cols_)
collisionsAndFinishes = pd.DataFrame(columns=cols,index=list(truearea.index))

for experiment in experiments:
	for map_ID in map_IDs:
		if data[experiment][map_ID]["number_of_collisions"] > 0 and not data[experiment][map_ID]["finished"]:
			collisionsAndFinishes[(experiment,"Both")][map_ID] = True
		if data[experiment][map_ID]["number_of_collisions"] > 0:
			collisionsAndFinishes[(experiment,"collision")][map_ID] = True
			collisionsAndFinishes[(experiment,"n_collision")][map_ID] = data[experiment][map_ID]["number_of_collisions"]
		if not data[experiment][map_ID]["finished"]:
			collisionsAndFinishes[(experiment,"notFinished")][map_ID] = True

collisionsAndFinishes.dropna(axis=0, how='all', inplace=True)
collisionsAndFinishes.fillna(value=False, inplace=True)
print(collisionsAndFinishes.sum(axis=0))

print("")
print("Mean exploration percentage")
percentage_explored_ = percentage_explored
percentage_explored_ = percentage_explored_.clip(upper=0.95)
print(percentage_explored_[experiments].mean())

#collisionsAndFinishes[collisionsAndFinishes[list(experiments)[1]]["notFinished"] == True].to_csv('RGS_did_not_finish.csv')
#collisionsAndFinishes[collisionsAndFinishes[list(experiments)[2]]["notFinished"] == True].to_csv('RGS_small_did_not_finish.csv')

print("")
print("Mean timesteps used  maps finished by all")
for experiment in experiments:
	#finished_map_IDs = list(collisionsAndFinishes[collisionsAndFinishes[experiment]["notFinished"]==False].index)
	#print(experiment + ":	" + str(timesteps[experiment].loc[finished_map_IDs].mean()))
	print(experiment + ":	" + str(timesteps[experiment].loc[map_IDs_finished_by_all].mean()))

print("")
print("Mean percentage explored for unfinished maps")
for experiment in experiments:
	unfinished_map_IDs = list(collisionsAndFinishes[collisionsAndFinishes[experiment]["notFinished"]==True].index)
	print(experiment + ":	" + str(percentage_explored_[experiment].loc[unfinished_map_IDs].mean()))

print("")
print("Collision pr. timesteps (permille)")
print(collisions.sum()/timesteps.sum()*1000)

print("")
#print(set(experiments.keys()))
for experiment in experiments.keys():
	experiments_ = list(experiments.keys()).copy()
	experiments_.remove(str(experiment))
	for experiment_ in experiments_:
		finished_map_IDs1 = list(collisionsAndFinishes[collisionsAndFinishes[experiment]["notFinished"]==False].index)
		finished_map_IDs2 = list(collisionsAndFinishes[collisionsAndFinishes[experiment_]["notFinished"]==False].index)
		n_maps_ = len(set(finished_map_IDs1) - set(finished_map_IDs2))
		print("maps finished by " + str(experiment) + " but not bys" + str(experiment_) + ": " + str(n_maps_))

bad_maps_for_RGS_small = list(percentage_explored[percentage_explored["damgaard2022RGS_small"]<0.6].index)
print("bad_maps_for_RGS_small: " + str(bad_maps_for_RGS_small))
bad_maps_for_RGS = list(collisions[collisions["damgaard2022RGS"]>0].index)
print("bad_maps_for_RGS: " + str(bad_maps_for_RGS))

# ############################## plot data ##############################
if not os.path.exists(tikzFolder):
    os.mkdir(tikzFolder)


# plt.figure()
# plt.title("Percentage Explored")
# plt.plot(range(len(truearea.index)), np.ones(len(truearea.index)), label="True", linewidth=linewidth, zorder = 2, color="black")
# plt.plot(range(len(truearea.index)), np.ones(len(truearea.index))*0.95, "--", linewidth=linewidth, label="95% True", zorder = 2, color="black")
# for key in percentage_explored.columns:
# 	if "_smoothed" in key:
# 		plt.plot(range(len(percentage_explored.index)), percentage_explored[key], linewidth=linewidth, label=key.replace("damgaard2022","").replace("_"," "), zorder = 2)
# 	else:
# 		plt.scatter(range(len(percentage_explored.index)), percentage_explored[key],alpha=alpha,label=key.replace("damgaard2022",""), zorder = 1)
# 		for map_ID in map_IDs:
# 			if data[key][map_ID]["number_of_collisions"] > 0:
# 				x = percentage_explored[key].index.get_loc(map_ID)
# 				y = percentage_explored[key][map_ID]
# 				plt.scatter(x, y, marker="x", color="black", label=None, zorder = linewidth)
# plt.legend()
# plt.xlabel("Map Index [n]")
# plt.ylabel(r"Percentage Explored [%]")

plt.figure()
title_str = "Percentage_Explored"
plt.title(title_str.replace("_",""))
plt.plot(truearea["m2"], np.ones(len(truearea.index)), label="True", linewidth=linewidth, zorder = 2, color=colors["black"])
plt.plot(truearea["m2"], np.ones(len(truearea.index))*0.95, "--", linewidth=linewidth, label="95% True", zorder = 2, color=colors["black"])
plt.plot([max_are_explored_with_backtracting, max_are_explored_with_backtracting], [0, 1], ":", linewidth=linewidth, zorder = 2, color=colors["black"])
#plt.plot([max_are_explored_without_backtracting, max_are_explored_without_backtracting], [0, 1], ":", linewidth=linewidth, zorder = 2, color=colors["black"])
for key in percentage_explored.columns:
	if "_smoothed" in key:
		plt.plot(truearea["m2"], percentage_explored[key], linewidth=linewidth, label=key.replace("damgaard2022","").replace("_"," "), zorder = 2, color=experiments[key.replace("_smoothed","")]["color"])
	else:
		plt.scatter(truearea["m2"], percentage_explored[key],alpha=alpha,label=key.replace("damgaard2022",""), zorder = 1, color=experiments[key]["color"])
		for map_ID in map_IDs:
			if data[key][map_ID]["number_of_collisions"] > 0:
				x = truearea["m2"][map_ID]
				y = percentage_explored[key][map_ID]
				#plt.scatter(x, y, marker="x", color=colors["black"], label=None, zorder = linewidth)
				plt.scatter(x, y, marker="x", color=experiments[key]["color"], label=None, zorder = linewidth)
plt.legend()
plt.xlabel(r"Map Size [$m^2$]")
plt.ylabel(r"Percentage Explored [\%]")
# tikz_file_path = tikzFolder + "/" + title_str + '.tikz'
# tikzplotlib.save(tikz_file_path)
# # modify generated tikz file
# width_str = "\\linewidth"
# plot_width = 1.0  # *width_str
# #plot_height = 0.64  # *width_str
# plot_height = 0.45  # *width_str
# fin = open(tikz_file_path, "rt")
# fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
# for line in fin:
#     if "\\begin{axis}[" in line:
#         fout.write(line)
#         fout.write("width={width}{width_str},\n".format(width=plot_width, width_str=width_str))
#         fout.write("height={height}{width_str},\n".format(height=plot_height,width_str=width_str))
#         fout.write("clip marker paths=true,\n")  # fixes error with the order of plotting
#         fout.write("scaled x ticks=false,\n")
#     elif "thick" in line:
#     	fout.write(line.replace("thick","very thick")) # line width!
#     elif "\\addlegendentry" in line and "%" in line:
#     	fout.write(line.replace("%",r"\%"))
#     else:
#         fout.write(line)
# fin.close()
# fout.close()
# os.remove(tikz_file_path)
# os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)


plt.figure()
#plt.title("Area Explored")
title_str = "Area_Explored"
plt.title(title_str.replace("_",""))
plt.plot(range(len(truearea.index)), truearea["m2"], label="True", linewidth=linewidth, zorder = 2, color=colors["black"])
plt.plot(range(len(truearea.index)), truearea["m2"]*0.95, "--", linewidth=linewidth, label="95% True", zorder = 2, color=colors["black"])
plt.plot([max_are_explored_with_backtracting_idx_number, max_are_explored_with_backtracting_idx_number], [0, np.max(truearea["m2"])], ":", linewidth=linewidth, zorder = 2, color=colors["black"])
#plt.plot([max_are_explored_without_backtracting_idx_number, max_are_explored_without_backtracting_idx_number], [0, np.max(truearea["m2"])], ":", linewidth=linewidth, zorder = 2, color=colors["black"])

for key in area_explored.columns:
	if "_smoothed" in key:
		plt.plot(range(len(area_explored.index)), area_explored[key], linewidth=linewidth, label=key.replace("damgaard2022","").replace("_"," "), zorder = 2, color=experiments[key.replace("_smoothed","")]["color"])
	else:
		idxNotFinished = list(collisionsAndFinishes[collisionsAndFinishes[key]["notFinished"]==True].index)
		idx_ = []
		for i in range(len(idxNotFinished)):
			idx_.append(area_explored[key].index.get_loc(idxNotFinished[i]))
		plt.scatter(idx_, area_explored[key][idxNotFinished],alpha=alpha,label=key.replace("damgaard2022","").replace("_"," ") + " not done", zorder = 1, color=experiments[key]["color"])
	#else:
	#	plt.scatter(range(len(area_explored.index)), area_explored[key],alpha=alpha,label=key.replace("damgaard2022","").replace("_"," "), zorder = 1, color=experiments[key]["color"])
plt.legend()
plt.xlabel("Map Index [n]")
plt.ylabel(r"Area Explored [$m^2$]")
tikz_file_path = tikzFolder + "/" + title_str + '.tikz'
tikzplotlib.save(tikz_file_path)
# modify generated tikz file
width_str = "\\linewidth"
plot_width = 1.0  # *width_str
plot_height = 0.64  # *width_str
fin = open(tikz_file_path, "rt")
fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
for line in fin:
    if "\\begin{axis}[" in line:
        fout.write(line)
        fout.write("width={width}{width_str},\n".format(width=plot_width, width_str=width_str))
        fout.write("height={height}{width_str},\n".format(height=plot_height,width_str=width_str))
        fout.write("clip marker paths=true,\n")  # fixes error with the order of plotting
        fout.write("scaled x ticks=false,\n")
    elif "thick" in line:
    #	fout.write(line.replace("thick","very thick")) # line width!
    #	fout.write(line.replace("very thick","ultra thick")) # line width!
    	fout.write(line.replace("very thick","line width=3pt"))
    elif "\\addlegendentry" in line and "%" in line:
    	fout.write(line.replace("%",r"\%"))
    #elif "nan" in line:
    #	fout.write(line.replace("nan",str(0)))
    else:
        fout.write(line)
fin.close()
fout.close()
os.remove(tikz_file_path)
os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)



plt.figure()
for experiment in experiments:
	#plt.scatter(truearea["n_rooms"], area_explored[experiment],alpha=1.0,label=experiment.replace("damgaard2022",""), zorder = 1, color=experiments[experiment]["color"])
	plt.scatter(area_explored[experiment], truearea["n_rooms"],alpha=1.0,label=experiment.replace("damgaard2022",""), zorder = 1, color=experiments[experiment]["color"])
	#plt.scatter(percentage_explored[experiment], truearea["n_rooms"],alpha=1.0,label=experiment.replace("damgaard2022",""), zorder = 1, color=experiments[experiment]["color"])
plt.legend()
plt.ylabel(r"Number of Rooms [n]")
plt.xlabel(r"Area Explored [$m^2$]")


plt.figure()
for experiment in experiments:
	plt.scatter(truearea["n_rooms"], timesteps[experiment],alpha=1.0,label=experiment.replace("damgaard2022",""), zorder = 1, color=experiments[experiment]["color"])
plt.legend()
plt.xlabel(r"Number of Rooms [n]")
plt.ylabel(r"Timesteps [$n$]")


# plt.figure()
# for experiment in experiments:
# 	finished_map_IDs = list(collisionsAndFinishes[collisionsAndFinishes[experiment]["notFinished"]==False].index)
#	plt.hist(timesteps[experiment].loc[finished_map_IDs],alpha=0.2, color=experiments[experiment]["color"])




plt.show()