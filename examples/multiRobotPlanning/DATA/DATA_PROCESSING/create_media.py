import imageio
import os


def main():
    deleteData = False
    speedup = 5
    FILE_TYPE = ".gif"
    #FILE_TYPE = ".mp4"
    save_folder = "../date_2022_01_27_time_13_06_03/MEDIA"
    
    for simID in range(29,33):
        simName = "pngs_sim_" + str(simID)
        MEDIA_data_folder = save_folder + "/" + simName
        FILE_NAME = simName + "_" + str(speedup) + "x"

        print("Starting Creating Gif for sim " + simName)

        with imageio.get_writer( save_folder + "/" + FILE_NAME + FILE_TYPE, mode='I', fps=5) as writer:
            N_frames = len(os.listdir(MEDIA_data_folder))
            for frame in range(0, N_frames, int(speedup)):
                try:
                    image = imageio.imread(MEDIA_data_folder + "/" + str(frame) + ".png")
                    writer.append_data(image)
                except:
                    print("Error with Frame: " + str(frame))
                print("Finished " + str(frame/N_frames*100) + " %")

        if deleteData:
            for f in os.listdir(MEDIA_data_folder):
                os.remove(os.path.join(MEDIA_data_folder, f))
            os.rmdir(MEDIA_data_folder)

        print("Finished Creating " + simName + FILE_TYPE)

if __name__ == '__main__':
    main()