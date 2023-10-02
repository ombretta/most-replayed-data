import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Add a command line argument for the file path
parser.add_argument('file_path', help='Path to the h5 file')

# Parse the command line arguments
args = parser.parse_args()

MAX_VIDEOS = 1
# Open the file using the file path passed as an argument
with h5py.File(args.file_path, 'r') as f:
    for k in list(f.keys())[:MAX_VIDEOS]:
        data = f[k]
        print(k)

        # Access the features 2D-array
        features = data['features']
        print(features.shape)

        # Access the gtscore 1D-array
        gtscore = np.array(data['gtscore'])
        print(gtscore.shape)

        # Access the user_summary 2D-array
        user_summary = np.array(data['user_summary'])
        print(user_summary.shape)

        # Plot the gtscore
        plt.plot(np.linspace(0, 1, gtscore.shape[0]), gtscore)
        plt.xlabel('Frame')
        plt.ylabel('gtscore')
            
        # # Plot the individual rows of user_summary
        for i in range(user_summary.shape[0]):
            plt.plot(np.linspace(0, 1, user_summary.shape[1]), user_summary[i])
            plt.xlabel('Frame')
            plt.ylabel('user_summary')
            # plt.title('User '+str(i))
        plt.show() 

        # # Access the other data in the 'key' group
        # change_points = data['change_points']
        # n_frame_per_seg = data['n_frame_per_seg']
        # n_frames = data['n_frames']
        # picks = data['picks']
        # n_steps = data['n_steps']
        # gtsummary = data['gtsummary']
