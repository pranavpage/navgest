import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
# Based on the trace, analyse the gesture 

# Actually, build a class with methods 
# Option 1 : analyse video stream and save the trace 
# Option 2 : playback the trace and analyse
# In each case, gesture detector runs 

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger
]

def get_vx_vy_landmark(df, landmark="WRIST", frame_num=1):
    # get the instantaneous velocity of the landmark in x,y coordinates for frame frame_num
    return

def det_wrist_flick():
    # detect a flick of the wrist (right hand outward)
    return

def play_trace(fname):
    df = pd.read_csv(fname)
    data = df.values  
    num_frames = len(df)

    fig, ax = plt.subplots()
    scatter, = ax.plot([], [], 'ro', markersize=5)

    def init():
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.set_aspect(9/16)
        return scatter,

    def update(frame):
        x = data[frame, 0::3]
        y = data[frame, 1::3]
        scatter.set_data(x, y)
        return scatter,

    ani = animation.FuncAnimation(fig, update, frames=num_frames,
                                  init_func=init, blit=True, interval = 50)

    plt.show()

    return ani

if __name__ == "__main__":
    play_trace("./data/default.csv")