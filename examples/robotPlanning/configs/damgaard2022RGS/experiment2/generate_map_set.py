# this file was used to generate the set of maps used for the second experiment in damgaard2022RGS
import os
import json

JSONfileDir = "/Users/mrd/Dropbox/PHD/Code/probMind/probMind/examples/misc/HouseExpo/HouseExpo/json"
damgaard2022AKS_fully_explored_map_IDs_file = "/Users/mrd/Dropbox/PHD/Code/probMind/probMind/examples/robotPlanning/configs/damgaard2022AKS/map_ids_fully_explored.txt"
new_file = "map_ids.txt"


min_room_num = 3
min_BB_size = 10*10


def list_files(dir):
    r = []
    for root, dirs, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def main():
	DATAfiles = list_files(JSONfileDir)
	jsonFiles = []
	for i in range(len(DATAfiles)):
	    if DATAfiles[i].endswith(".json"):
	        jsonFiles.append(DATAfiles[i])

	with open(damgaard2022AKS_fully_explored_map_IDs_file) as file:
	    map_IDs = file.readlines()
	map_IDs = [map_ID.rstrip() for map_ID in map_IDs] # remove linebreak character


	print("N maps finished: " + str(len(map_IDs)))


	map_IDs_to_save = []
	map_BB_sizes = []

	for map_ID in map_IDs:
		jsonFile = [string for string in jsonFiles if map_ID in string]
		with open(jsonFile[0], 'rb') as f_json:
			json_data_map_i = json.load(f_json)
			if map_ID == map_IDs[0]:
				print("Available keys")
				print(json_data_map_i.keys())
			bb_size = ((json_data_map_i["bbox"]["max"][0]-json_data_map_i["bbox"]["min"][0])*
					   (json_data_map_i["bbox"]["max"][1]-json_data_map_i["bbox"]["min"][1]))
			if json_data_map_i['room_num'] >= min_room_num and bb_size >= min_BB_size:
				map_IDs_to_save.append(map_ID)

	print("maps in new set: " + str(len(map_IDs_to_save)))

	with open(new_file, 'w') as f:
		for i in range(len(map_IDs_to_save)):
			f.write(map_IDs_to_save[i] + "\n")

def redefine_map_set():
	old_map_IDs_file = "map_ids.txt" 
	DATA_FOLDER_ = "../../../DATA"
	DATA_FOLDER = DATA_FOLDER_ + "/" + "date_2022_09_24_time_08_14_15_thread_IDs_0_2"

	files_ = list_files(DATA_FOLDER)
	map_IDs_done = []
	for file in files_:
		if file.endswith(".xz"):
			map_ID = file.replace(DATA_FOLDER+"/","").partition("/")[0]
			map_IDs_done.append(map_ID)

	fin = open(old_map_IDs_file, "rt")
	map_IDs_to_do = []
	for map_ID_ in fin:
			map_IDs_to_do.append(map_ID_.replace("\n",""))

	map_IDs_remaining = list(set(map_IDs_to_do)-set(map_IDs_done))
	print(len(map_IDs_to_do))
	print(len(map_IDs_done))
	print(len(map_IDs_remaining))

	with open(old_map_IDs_file.replace("ids","ids_remaining"), "wt") as fout:
		for map_ID_ in map_IDs_remaining:
			fout.write(map_ID_+"\n")


if __name__ == '__main__':
    #main()
    redefine_map_set()