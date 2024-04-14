import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
import cv2, sched, time
import mediapipe as mp 

# Making a class to handle everything 
# When navgest.per_frame() is called, depending on the mode, either the current frame from the video capture
# is processed by the model and the landmarks are output, or the landmarks from a stored .csv trace 
# are output 
class navgest:
    def __init__(self, mode : str, save_trace : bool, interval : float):
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
        self.last_state = self.state
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
        return
    
    def start(self):
        self.per_frame()
        self.scheduler.enterabs(time.time() + self.interval, 1, self.repeat, ())
        self.scheduler.run()
        return
    
    def repeat(self):
        next_time = time.time() + self.interval
        self.scheduler.enterabs(next_time, 1, self.repeat, ())
        self.per_frame()

        if cv2.waitKey(1) & 0xFF == ord('q'):
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
                            self.process_state()

                cv2.imshow("Live Stream", frame)

            else:
                print("FAIL")
        return
    
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

    def process_state(self):
        # for now, just print velocity
        self.track_wrist()
        return
    
    def track_wrist(self):
        diff = (self.state - self.last_state)/self.interval
        print(f"Wrist vx, vy = {diff[:2]}")
        return
    
def main():
    nv = navgest("live", False, 0.1)
    nv.start()
if __name__ == "__main__":
    main()
    