import pickle
import lzma
import os
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
import math
import imageio
import pandas as pd

import matplotlib as mpl
#mpl.rcParams.update(mpl.rcParamsDefault)

import tikzplotlib

# latex formatting strings:
# https://matplotlib.org/stable/tutorials/text/usetex.html
# https://matplotlib.org/stable/gallery/text_labels_and_annotations/tex_demo.html
# remember that special charathers in strings in figures have to have prober escaping: i.e. use "\%" instead of "%"
plt.rcParams['text.usetex'] = True

robot_radius = 0.35
goal_radius = 0.5
alpha = 0.8
scale = 1/10
fadeLength = 75
markerSize = 5
fadeLength = 50
stepIntervals = [0, 30, 60, 100]
#stepIntervals = [0, 30, 60, 110]

DATA_FOLDER = "DATA"
tikzFolder = "tikzPlots"
experiments = {"V_shape": {"dataFolder": 	"date_2022_09_20_time_08_02_43_thread_IDs_0_2", 
						   #"intervals": 	[0, 30, 40.5, 100, 115], 
						   "intervals": 	[0, 37.5, 100, 115], 
						   "range": 		[22.5,115],
						   "stepIntervals": stepIntervals,
						   "goalPosition": 	[15.0, 7.5]
						   },
		       "C_shape": {"dataFolder": 	"date_2022_09_20_time_14_09_10_thread_IDs_0_2", 
		       			   "intervals": 	[0, 55, 75], 
		       			   "range": 		[37.5,75],
		       			   "stepIntervals": stepIntervals,
						   "goalPosition": 	[16.75, 7.5],
		       			   },
		       "double_U_Shape": {"dataFolder": 	"date_2022_09_20_time_19_07_11_thread_IDs_0_2", 
		                		  #"intervals": 		[0, 42.5, 44.0, 75], 
		                		  "intervals": 		[0, 41.2, 55, 75], 
		                		  #"intervals": 		[0, 41.2, 75], 
		                		  "range": 			[27.5,75],
		                		  "stepIntervals": stepIntervals,
								  "goalPosition": 	[11.5, 12.5]}
		       }

experiments_ = ["V_shape", "C_shape"]
experiments_ = list(experiments.keys())

colors = {
    "yellow": "#FBBC05",
    "green": "#34A853",
    "red": "#EA4335",
    "blue": "#4285F4",
    "purple": "#410093",
    "pink": "#FF33FF",
    "oliveGreen": "#AFB83B",
    "black": "#000001",
}


def colorFader_(c1, c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c1=np.array(c1)
    c2=np.array(c2)
    return (1-mix)*c1 + mix*c2

def colorFader(c1, c2, L, mixType="linear"): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
    c = []
    for l in range(L):
	    if mixType=="linear":
	    	c.append(colorFader_(c1, c2, mix=l/L))
	    elif mixType=="exp":
	    	mix = np.exp(-5*(L-l)/L)
	    	c.append(colorFader_(c1, c2, mix=mix))
    return c


if not os.path.exists(tikzFolder):
    os.mkdir(tikzFolder)

def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r


for experiment in experiments_:
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

	print("Loading data for experiment: " + str(experiment))
	DATA_FOLDER_ = DATA_FOLDER + "/" + experiments[experiment]["dataFolder"]

	files_ = list_files(DATA_FOLDER_)
	#files_ = files_[0:30]
	if experiment not in data:
		data[experiment] = {}

	counter = 0
	n_files = len(files_)
	for file in files_:
		counter = counter + 1
		if file.endswith(".xz"):
			idx = file.replace(DATA_FOLDER_+"/"+experiment+"/","").replace(".xz","").replace("/","_")
			if idx not in data[experiment]:
				f = lzma.open(file, 'rb')
				data_ = pickle.load(f)

				data[experiment][idx] = {}
				data[experiment][idx]["trajectory"] = data_["trajectory"]
				dist = 0
				for tau in range(1,len(data_["trajectory"])):
					step_distance = np.sqrt(	(data_["trajectory"][tau][0]-data_["trajectory"][tau-1][0])**2 +
									 			(data_["trajectory"][tau][1]-data_["trajectory"][tau-1][1])**2)
					dist = dist + step_distance

				data[experiment][idx]["travelled_dist"] = dist
				if "trueMap" in data_:
					data[experiment]["trueMap"] = data_["trueMap"]
					data[experiment]["meter2pixel"] = data_["meter2pixel"]

		if counter % int(n_files/10) == 0:
			percentageLoaded = counter/len(files_)*100
			print("Loaded {percentageLoaded:.2f} %".format(percentageLoaded=percentageLoaded))

	with lzma.open(resultsDATAfile, "wb") as f:
	    pickle.dump(data, f)


# make plot
U = len(experiments[experiments_[0]]["stepIntervals"])
nrows = U+1
fig, axs = plt.subplots(nrows=nrows, ncols=len(experiments), figsize=(3*24*scale,nrows*15*scale))

for k in range(len(experiments_)):
	for u in range(0,U):
		if k == 0:
			axs[u, k].set(ylabel='y-coordinate [m]')
		if k == len(experiments_)-1 and u<U-1:
			tau2_ = experiments[experiments_[k]]["stepIntervals"][u+1]
			tau1_ = np.max([0,tau2_-fadeLength])
			axs[u, k].set(ylabel=r'$t\in['+str(tau1_)+r';'+str(tau2_)+r']$')
			#axs[u, k].set(ylabel=r'$t\in[T_{max}-' + str(fadeLength) + r';T_{max}]$')
			axs[u, k].yaxis.set_label_position('right') 
		if k == len(experiments_)-1 and u==U-1:
			axs[u, k].set(ylabel=r'$t\in[T_{max}-' + str(fadeLength) + r';T_{max}]$')
			axs[u, k].yaxis.set_label_position('right') 
		if u == 0:
			axs[u, k].set_title(experiments_[k].replace("_"," "))
			#axs[u, k].set(xlabel='x-coordinate [m]')
			#axs[u, k].xaxis.set_label_position('top') 
		if u == U-1:
			axs[u, k].set(xlabel='x-coordinate [m]')

		data_ = data[experiments_[k]].copy()
		del data_["trueMap"]
		del data_["meter2pixel"]
		pd_ = pd.DataFrame.from_dict(data_, orient='index')
		#pd_ = pd_[:][list(pd_.index[0:3])]

		range_ = experiments[experiments_[k]]["range"]
		n_bins = int((range_[1]-range_[0])/2.5)
		counts, bins = np.histogram(pd_["travelled_dist"], range=range_, bins = n_bins)

		intervals = experiments[experiments_[k]]["intervals"]
		interval_idx = []
		bin_indexes = []
		for i in range(1,len(intervals)):
			interval_idx_1 = set(pd_[pd_["travelled_dist"]>=intervals[i-1]].index)
			interval_idx_2 = set(pd_[pd_["travelled_dist"]<intervals[i]].index)
			interval_idx_ = list(interval_idx_2.intersection(interval_idx_1))
			interval_idx.append(interval_idx_)

			#bin_indexes_ = [n for n, el in enumerate(bins) if (el>=intervals[i-1] and el<intervals[i])]
			#if i < len(intervals)-1:
			#	bin_indexes_ = np.append(bin_indexes_, bin_indexes_[-1]+1)
			#bin_indexes.append(bin_indexes_)

		map_grid = data[experiments_[k]]["trueMap"]
		meter2pixel = data[experiments_[k]]["meter2pixel"]
		mapShape = map_grid.shape
		axs[u,k].imshow(map_grid, cmap='binary', origin="lower", extent=[0, mapShape[1] / meter2pixel, 0, mapShape[0] / meter2pixel], aspect="auto", vmin=0.0, vmax=1.0)
		axs[u,k].set(adjustable="datalim")

		x = data_[interval_idx[0][0]]["trajectory"][0,0]
		y = data_[interval_idx[0][0]]["trajectory"][0,1]
		if k == len(experiments_)-1:
			goalZone = plt.Circle((experiments[experiments_[k]]["goalPosition"][0], experiments[experiments_[k]]["goalPosition"][1]), goal_radius, fill=True, edgecolor=None, alpha=0.7, color=colors["purple"], zorder=101, label=r"Goal Zone")
			initial_pos = plt.Circle((x, y), robot_radius, color=colors["blue"], fill=True, edgecolor=None, alpha=0.7, zorder=101, label=r"Initial Position")
		else:
			goalZone = plt.Circle((experiments[experiments_[k]]["goalPosition"][0], experiments[experiments_[k]]["goalPosition"][1]), goal_radius, fill=True, edgecolor=None, alpha=0.7, color=colors["purple"], zorder=101)
			initial_pos = plt.Circle((x, y), robot_radius, color=colors["blue"], fill=True, edgecolor=None, alpha=0.7, zorder=101)
		axs[u,k].add_patch(goalZone)
		axs[u,k].add_patch(initial_pos)
		
		max_length = 0
		for j in range(len(interval_idx)):
			for idx in interval_idx[j]: 
				if len(data_[idx]["trajectory"]) > max_length:
					max_length = len(data_[idx]["trajectory"])

		print("max_length of map " + experiments_[k] + " is: " + str(max_length))

		if u < len(experiments[experiments_[k]]["stepIntervals"])-1:
			stepInterval_ = experiments[experiments_[k]]["stepIntervals"][u+1]
		else:
			stepInterval_ = max_length
		for j in range(len(interval_idx)):
			for idx in interval_idx[j]:
				if stepInterval_<len(data_[idx]["trajectory"])-1:
					fadeLength_ = np.max([0,stepInterval_-fadeLength])
					x = data_[idx]["trajectory"][fadeLength_:stepInterval_,0]
					y = data_[idx]["trajectory"][fadeLength_:stepInterval_,1]
				else:
					fadeLength_ = fadeLength-(stepInterval_-len(data_[idx]["trajectory"]))
					fadeLength_ = np.min([len(data_[idx]["trajectory"]),fadeLength_])
					x = data_[idx]["trajectory"][len(data_[idx]["trajectory"])-fadeLength_:,0]
					y = data_[idx]["trajectory"][len(data_[idx]["trajectory"])-fadeLength_:,1]		

				# color_ = list(colors.values())[j]
				# Tau = len(x)
				# for tau in range(Tau):
				# 	if j == len(interval_idx)-1 and idx == interval_idx[j][-1] and k == len(experiments_)-1 and tau==Tau-1:
				# 		trajectory_line, = axs[u,k].plot(x[tau:tau+2],y[tau:tau+2], color=color_, alpha=np.exp(-7*(Tau-tau)/Tau), zorder=100-len(interval_idx[j]), label = r"Trajectories")
				# 	else:
				# 		trajectory_line, = axs[u,k].plot(x[tau:tau+2],y[tau:tau+2], color=color_, alpha=np.exp(-7*(Tau-tau)/Tau), zorder=100-len(interval_idx[j]))
								
				# if stepInterval_<len(data_[idx]["trajectory"])-1:
				# 	x = data_[idx]["trajectory"][0:stepInterval_,0]
				# 	y = data_[idx]["trajectory"][0:stepInterval_,1]
				# else:
				# 	x = data_[idx]["trajectory"][:,0]
				# 	y = data_[idx]["trajectory"][:,1]	

				color_ = list(colors.values())[j]					
				if j == len(interval_idx)-1 and idx == interval_idx[j][-1] and k == len(experiments_)-1:
					trajectory_line, = axs[u,k].plot(x,y, color=color_, alpha=alpha, zorder=100-len(interval_idx[j]), label = r"Trajectories")
				else:
					trajectory_line, = axs[u,k].plot(x,y, color=color_, alpha=alpha, zorder=100-len(interval_idx[j]))
				

				# c1 = mpl.colors.to_rgba(list(colors.values())[j], alpha=0.0)
				# c2 = mpl.colors.to_rgba(list(colors.values())[j], alpha=1.0)
				# colors_ = colorFader(c1, c2, len(x), mixType="exp")
				# if j == len(interval_idx)-1 and idx == interval_idx[j][-1] and k == len(experiments_)-1:
				# 	axs[u,k].scatter(x,y, s=markerSize, c=colors_, zorder=100-len(interval_idx[j]), label = r"Trajectories")
				# else:					
				# 	axs[u,k].scatter(x,y, s=markerSize, c=colors_, zorder=100-len(interval_idx[j]))

	# histogram plots
	max_ = 0
	for j in range(len(interval_idx)):
		counts_, bins_ = np.histogram(pd_["travelled_dist"][interval_idx[j]], range=range_, bins = n_bins)
		n_, bins_, patches_ = axs[axs.shape[0]-1,k].hist(bins_[:-1], bins_, weights=counts_, color=list(colors.values())[j], align="mid", alpha = 1.0)
		if np.max(n_) > max_:
			max_ = np.max(n_)

	if k == 0:
		axs[axs.shape[0]-1, k].set(ylabel='Count [n]')
	axs[axs.shape[0]-1, k].set(xlabel='Travelled Distance[m]')

	for j in range(1,len(intervals)-1):
		if j == len(intervals)-2 and k == len(experiments_)-1:
			intervals_line, = axs[axs.shape[0]-1,k].plot([intervals[j], intervals[j]],[0, max_], "--", color=colors["black"], label=r"Intervals", alpha=1.0) # we need to specify opacity/alpha to make legend work
		else:
			intervals_line, = axs[axs.shape[0]-1,k].plot([intervals[j], intervals[j]],[0, max_], "--", color=colors["black"])



#initial_pos_label = r"Initial Position"
#goalZone_label = r"Goal Zone"
#intervals_line_label = r"Intervals"
#trajectory_line_label = r"Trajectories"
#axs[axs.shape[0]-1,axs.shape[1]-1].legend([initial_pos, goalZone, intervals_line, trajectory_line], [initial_pos_label, goalZone_label, intervals_line_label, trajectory_line_label])
#axs[0,1].legend([initial_pos, goalZone, intervals_line, trajectory_line], [initial_pos_label, goalZone_label, intervals_line_label, trajectory_line_label])
#axs[0,1].legend()
#handles, labels = axs[0,1].get_legend_handles_labels()
#print(handles)

handles1, labels1 = axs[0,len(experiments_)-1].get_legend_handles_labels()
handles2, labels2 = axs[axs.shape[0]-1,len(experiments_)-1].get_legend_handles_labels()
print(labels1)
for label in labels2:
	labels1.append(label)
for handle in handles2:
	handles1.append(handle)
print(handles1)
axs[0,len(experiments_)-1].legend(handles1, labels1, loc="lower right", ncol=len(labels1))
#axs[axs.shape[0]-1,len(experiments_)-1].legend(handles1, labels1, loc="lower right", ncol=len(labels1))
axs[axs.shape[0]-1,0].legend(handles1, labels1, loc="lower right", ncol=len(labels1))


title_str = "experiment1"
tikz_file_path = tikzFolder + "/" + title_str + '.tikz'
tikzplotlib.save(tikz_file_path)
# modify generated tikz file
width_str = "\\linewidth"
width_str_scale = 0.95
plots_pr_row = axs.shape[1]
rows = axs.shape[0]
vertical_sep = width_str_scale*0.05  # *width_str
vertical_sep2 = width_str_scale*0.1  # *width_str
vertical_sep_last_row = width_str_scale*0.5 # in cm
plot_widths = width_str_scale*(1-vertical_sep2*(plots_pr_row-1))/plots_pr_row  # *width_str
plot_heights = plot_widths*(15/24)
relative_path = "figures/"
string_to_replace = "experiment1"
fin = open(tikz_file_path, "rt")
fout = open(tikzFolder + "/tmp_tikz_file.tikz", "wt")
CommonLegend_counter = 0
legend_fill_opacity_str = ""
legend_draw_opacity_str = ""
for line in fin:
    if "group style" in line:  # replace group plot args to fit text width
        fout.write("\\begin{{groupplot}}[group style={{group size={coloumns} by {rows},vertical sep={vertical_sep}{width_str}}},scale only axis, width={width}{width_str}, height={height}{width_str}] \n".format(groupplot="groupplot",coloumns=plots_pr_row,rows=rows,width=plot_widths, height=plot_heights,vertical_sep=vertical_sep,width_str=width_str)) 
    elif "legend cell align" in line:  # add legend name
        fout.write("legend to name=CommonLegend" + str(CommonLegend_counter) + "," + "\n")
        fout.write(line)
        CommonLegend_counter = CommonLegend_counter + 1
        if CommonLegend_counter == 2:
        	fout.write("yshift=-{vertical_sep}cm".format(vertical_sep=vertical_sep_last_row) + "," + "\n") # <-- to make space for x-axis-label
    elif "nextgroupplot" in line:
        if CommonLegend_counter == 2:
        	fout.write(line)
        	fout.write("yshift=-{vertical_sep}cm".format(vertical_sep=vertical_sep_last_row) + "," + "\n") # <-- to make space for x-axis-label
        else:
        	fout.write(line)
    elif "legend style={" in line:  # add more space between legend entries
        fout.write(line)
        fout.write("  /tikz/every even column/.append style={column sep=0.5cm},\n")
    elif "\\addplot graphics" in line:  # remove legend from included graphics
        tmp_line = line.replace(string_to_replace, relative_path+string_to_replace)  # replace relative paths to other input files
        fout.write(tmp_line.replace("\\addplot graphics", "\\addplot [forget plot] graphics"))
    elif "fill opacity" in line:
    	legend_fill_opacity_str = line.replace("\n","")
    	fout.write("  fill opacity=0.0,\n")
    elif "draw opacity" in line:
    	legend_draw_opacity_str = line.replace("\n","")
    	fout.write("  draw opacity=0.0,\n")
    elif "\\end{tikzpicture}" in line:  # specify legend placement below groupplot
        fout.write("\\coordinate (c3) at ($(current bounding box.south east)!.5!(current bounding box.south west)$);" + "\n")
        fout.write("\\node[below,{legend_fill_opacity_str}{legend_draw_opacity_str} draw=white!80!black] at (c3 |- current bounding box.south){{".format(legend_fill_opacity_str=legend_fill_opacity_str,legend_draw_opacity_str=legend_draw_opacity_str))
        for n in range(CommonLegend_counter):
        	fout.write("\\pgfplotslegendfromname{CommonLegend" + str(n) +"}")
        fout.write("};" + "\n")
        fout.write(line)
    elif "ylabel=" in line and "t\in" in line:
    	fout.write(line)
    	fout.write("ylabel style={{yshift=-{y_shift}\\linewidth}},".format(y_shift=plot_widths+0.07))

    #elif "legend columns" in line: # limit the number of columns in the legend
    #        fout.write("legend columns=2," + "\n")
    #elif "thick" in line:
    #	fout.write(line.replace("thick","very thick")) # line width!
    #	fout.write(line.replace("very thick","ultra thick")) # line width!
    #	fout.write(line.replace("very thick","line width=3pt"))
    else:
        fout.write(line)
fin.close()
fout.close()
os.remove(tikz_file_path)
os.rename(tikzFolder + "/tmp_tikz_file.tikz", tikz_file_path)


plt.show()