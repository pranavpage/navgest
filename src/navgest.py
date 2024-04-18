import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import cv2, sched, time, csv, pyautogui
import mediapipe as mp 

# Making a class to handle everything 
# When navgest.per_frame() is called, depending on the mode, either the current frame from the video capture
# is processed by the model and the landmarks are output, or the landmarks from a stored .csv trace 
# are output 


# Class to store average velocity, total displacement of a landmark
class accumulate_motion:
    def __init__(self, landmark):
        self.average_vx = 0
        self.average_vy = 0
        self.del_x = 0
        self.del_y = 0
        self.time = 0
        self.landmark = landmark
        pass

    def __repr__(self) -> str:
        return f"Accumulator for {self.landmark}, time={self.time}, displacement since init = ({self.del_x:.3f}, {self.del_y:.3f}), avg velocity = {self.average_vx:.3f}, {self.average_vy:.3f}"

    def update(self, vx, vy, interval):
        self.del_x += interval*vx
        self.del_y += interval*vy
        self.time += interval
        self.average_vx = (self.del_x)/self.time 
        self.average_vy = (self.del_y)/self.time
        pass

    def reset(self):
        self.average_vx = 0
        self.average_vy = 0
        self.del_x = 0
        self.del_y = 0
        self.time = 0
        pass 
        
class navgest:
    def __init__(self, mode : str, save_trace : bool, interval : float, trace_file : str = None):
        self.mode = mode 
        self.save_trace = save_trace
        self.interval = interval
        self.cap = None 
        self.scheduler = sched.scheduler(time.time, time.sleep)
        mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)
        self.state = np.empty(63)
        self.state[:] = np.nan
        self.last_timestamp = np.nan
        self.last_state = self.state
        self.diff = self.state
        self.trace_file = trace_file

        self.landmark_indices = {
            'WRIST': 0,
            'THUMB_CMC': 1,
            'THUMB_MCP': 2,
            'THUMB_IP': 3,
            'THUMB_TIP': 4,
            'INDEX_FINGER_MCP': 5,
            'INDEX_FINGER_PIP': 6,
            'INDEX_FINGER_DIP': 7,
            'INDEX_FINGER_TIP': 8,
            'MIDDLE_FINGER_MCP': 9,
            'MIDDLE_FINGER_PIP': 10,
            'MIDDLE_FINGER_DIP': 11,
            'MIDDLE_FINGER_TIP': 12,
            'RING_FINGER_MCP': 13,
            'RING_FINGER_PIP': 14,
            'RING_FINGER_DIP': 15,
            'RING_FINGER_TIP': 16,
            'PINKY_MCP': 17,
            'PINKY_PIP': 18,
            'PINKY_DIP': 19,
            'PINKY_TIP': 20
        }

        if self.mode == "live":
            self.cap = cv2.VideoCapture(0)
            if(self.save_trace):
                self.make_trace_file()
        elif self.mode == "playback":
            self.trace_gen = self.trace_generator()
        return
    
    def make_trace_file(self):
        with open(self.trace_file, 'w', newline='') as file:
            writer = csv.writer(file)
            headers = []
            for name in self.landmark_indices.keys():
                headers += [f'{name}_x', f'{name}_y', f'{name}_z']
            headers += ['time']
            writer.writerow(headers)
        return
    
    def trace_generator(self):
        data = pd.read_csv(self.trace_file)
        for _, row in data.iterrows():
            timestamp = row.get('time', None)  # Fetch timestamp if available
            landmarks_vector = row.drop('time', errors='ignore').values
            yield (timestamp, landmarks_vector)

    def start(self):
        self.start_time = time.time()
        self.per_frame()
        self.scheduler.enterabs(time.time() + self.interval, 1, self.repeat, ())
        self.scheduler.run()
        return
    
    def repeat(self):
        next_time = time.time() + self.interval
        self.scheduler.enterabs(next_time, 1, self.repeat, ())
        ret = self.per_frame()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
            self.stop()
    
    def per_frame(self):
        if(self.mode == "live") and (self.cap.isOpened()):
            # Get the current frame
            success, frame = self.cap.read()
            if(success):
                rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == 'Right':

                            landmark_vector = self.landmarks_to_vector(hand_landmarks)
                            # print("Landmark vector:", landmark_vector)
                            self.last_state = self.state
                            self.state = landmark_vector
                            self.diff = (self.state - self.last_state)/self.interval
                            self.trace_state()
                            self.process_state()
                            

                cv2.imshow("Live Stream", frame)
                return 1

            else:
                print("FAIL")
                return 1
        elif(self.mode == "playback"):
            # play the trace from the file 
            try:
                timestamp, landmark_vector = next(self.trace_gen)                
                self.last_state = self.state
                self.state = landmark_vector
                if(self.last_timestamp):
                    self.diff = (self.state - self.last_state)/(timestamp - self.last_timestamp)
                self.last_timestamp = timestamp
                self.process_state()
                return 1
            except StopIteration:
                print("Finished playback.")
                return None            
        return
    
    def trace_state(self):
        with open(self.trace_file, 'a', newline='') as file:
            writer = csv.writer(file)
            row = list(self.state) + [time.time()]
            writer.writerow(row)
    
    def landmarks_to_vector(self, landmarks):
        if(landmarks):
            landmark_flat_vec = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z] 
                                for i in range(21)]).flatten()
        else:
            landmark_flat_vec = np.empty(63)
            landmark_flat_vec[:] = np.nan 
        return landmark_flat_vec
    
    def stop(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.scheduler.cancel(self.scheduler.queue[0])
   
    def __del__(self):
        self.stop()

    def get_xy_from_vector(self, vector, landmark_name):
        if landmark_name in self.landmark_indices:
            base_index = 3 * self.landmark_indices[landmark_name]
            x = vector[base_index]
            y = vector[base_index + 1]
            return (x, y)
        else:
            raise ValueError("Invalid landmark name provided.")


    def process_state(self):
        # for now, just print velocity
        self.track_points()
        self.det_wrist_flick()
        return
    
    def det_wrist_flick(self):
        # wrist might not move, or move slightly 
        # Index and Pinky might move fast and more than wrist

        # let's try mag(vx,vy) > 0.2 for 0.4 s
        # and net delta x > 0.1, net delta y < 0
        # above for both index and pinky 
        # z increases and then decreases?
        index_vx, index_vy = self.get_xy_from_vector(self.diff, "INDEX_FINGER_TIP")
        try:
            mag_v = (index_vx**2 + index_vy**2)**(0.5)
            if(mag_v > 0.75 and index_vx > 0.2):
                self.IFT.update(index_vx, index_vy, self.interval)
            if((abs(self.IFT.average_vx) > 1) and (abs(self.IFT.average_vy) > 0.8)):
                print("\n" + "-"*20 + "Wrist Flick" + "-"*20 + "\n")
                self.IFT.reset()
                self.action_wrist_flick()
        except AttributeError:
            self.IFT = accumulate_motion("INDEX_FINGER_TIP")
        return

    def track_points(self):
        
        # print("Wrist vx, vy = ", self.get_xy_from_vector(self.diff, "WRIST"))
        # print("Index tip vx, vy = ", self.get_xy_from_vector(self.diff, "INDEX_FINGER_TIP"))
        # print("Pinky tip vx, vy = ", self.get_xy_from_vector(self.diff, "PINKY_TIP"))
        return
    
    def action_wrist_flick(self):
        try:
            pyautogui.keyDown('ctrl')
            time.sleep(0.1)
            pyautogui.press('tab')
            time.sleep(0.1)
            pyautogui.keyUp('ctrl')
        except Exception as e:
            print(e)
        return

    
def main():
    nv = navgest("live", True, 0.1, "./data/default.csv")
    nv.start()
if __name__ == "__main__":
    main()
    