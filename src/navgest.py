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


def print_log(event):
    print('-'*30 + f' {event} ' + '-'*30 + '\n')
    return

# Class to store average velocity, total displacement of a landmark
class accumulate_motion:
    def __init__(self, landmark, v_th):
        self.average_vx = 0
        self.average_vy = 0
        self.del_x = 0
        self.del_y = 0
        self.time_acc = 0
        self.landmark = landmark
        self.v_th = v_th
        self.fist_counter = 0

    def __repr__(self) -> str:
        return f"Accumulator for {self.landmark},\
    time={self.time_acc}, displacement since init = ({self.del_x:.3f},\
    {self.del_y:.3f}),\
    avg velocity = {self.average_vx:.3f}, {self.average_vy:.3f}, \
    counter = {self.fist_counter}"

    def update(self, vx, vy, interval):
        mag_v = (vx**2 + vy**2)**0.5
        self.fist_counter -= interval
        if (self.fist_counter < 0):
            self.reset()
        elif (mag_v >= self.v_th):
            self.del_x += interval*vx
            self.del_y += interval*vy
            self.time_acc += interval
            self.average_vx = (self.del_x)/self.time_acc
            self.average_vy = (self.del_y)/self.time_acc

    def reset(self):
        self.average_vx = 0
        self.average_vy = 0
        self.del_x = 0
        self.del_y = 0
        self.time_acc = 0
        self.fist_counter = 0


class navgest:
    def __init__(self, mode: str, save_trace: bool, interval: float, trace_file: str = None):
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
        self.IFT = accumulate_motion("INDEX_FINGER_TIP", 0.5)
        self.TT = accumulate_motion("THUMB_TIP", 0.5)
        self.MFT = accumulate_motion("MIDDLE_FINGER_TIP", 1)
        self.RFT = accumulate_motion("RING_FINGER_TIP", 1)
        self.PT = accumulate_motion("PINKY_TIP", 1)

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
        self.scheduler.enterabs(time.time() + self.interval,
                                1, self.repeat, ())
        self.scheduler.run()
        return

    def repeat(self):
        next_time = time.time() + self.interval
        self.scheduler.enterabs(next_time, 1, self.repeat, ())
        ret = self.per_frame()

        if (cv2.waitKey(1) & 0xFF == ord('q')) or (not ret):
            self.stop()

    def per_frame(self):
        if (self.mode == "live") and (self.cap.isOpened()):
            # Get the current frame
            success, frame = self.cap.read()
            if (success):
                rgb_frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                results = self.hands.process(rgb_frame)
                if results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == 'Right':
                            landmark_vector = self.landmarks_to_vector(hand_landmarks)
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
        elif (self.mode == "playback"):
            # play the trace from the file
            try:
                timestamp, landmark_vector = next(self.trace_gen)
                self.last_state = self.state
                self.state = landmark_vector
                if (self.last_timestamp):
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
        if (landmarks):
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
        self.det_closed_fist()
        self.track_tips()
        self.det_wrist_flick()
        self.det_swipe_up()
        self.det_swipe_down()
        # self.det_italian() # italian hand gesture?
        # self.det_namaste() # Open/Close
        return

    def det_closed_fist(self):
        # detect closed fist

        # average distance between 4 fingers and thumb < threshold
        index_x, index_y = self.get_xy_from_vector(self.state, self.IFT.landmark)
        pinky_x, pinky_y = self.get_xy_from_vector(self.state, self.PT.landmark)
        thumb_x, thumb_y = self.get_xy_from_vector(self.state, self.TT.landmark)
        index_dist = ((thumb_x - index_x)**2 + (thumb_y - index_y)**2)**0.5
        pinky_dist = ((thumb_x - pinky_x)**2 + (thumb_y - pinky_y)**2)**0.5
        if (index_dist < 0.1 and pinky_dist < 0.15):
            # print_log('Closed Fist')
            self.IFT.fist_counter = 5*self.interval
            self.TT.fist_counter = 5*self.interval
            self.MFT.fist_counter = 5*self.interval
            self.RFT.fist_counter = 5*self.interval
            self.PT.fist_counter = 5*self.interval
        return

    def track_tips(self):
        # Init accumulators for tips of fingers 
        index_vx, index_vy = self.get_xy_from_vector(self.diff, self.IFT.landmark)
        thumb_vx, thumb_vy = self.get_xy_from_vector(self.diff, self.TT.landmark)
        middle_vx, middle_vy = self.get_xy_from_vector(self.diff, self.MFT.landmark)
        ring_vx, ring_vy = self.get_xy_from_vector(self.diff, self.RFT.landmark)
        pinky_vx, pinky_vy = self.get_xy_from_vector(self.diff, self.PT.landmark)
        self.IFT.update(index_vx, index_vy, self.interval)
        self.TT.update(thumb_vx, thumb_vy, self.interval)
        self.MFT.update(middle_vx, middle_vy, self.interval)
        self.RFT.update(ring_vx, ring_vy, self.interval)
        self.PT.update(pinky_vx, pinky_vy, self.interval)
        return

    def det_wrist_flick(self):
        # wrist might not move, or move slightly
        # Index and Pinky might move fast and more than wrist
        # print(self.IFT)
        if (self.IFT.average_vx > 0.5 and self.IFT.time_acc > self.interval):
            print_log('Wrist Flick')
            self.IFT.reset()
        return


    def det_swipe_up(self):
        # thumb doesn't move much
        # all four fingers move up fast
        if (self.IFT.average_vy < -0.4 and self.IFT.time_acc > self.interval):
            print_log('Swipe Up')
            self.IFT.reset()
        return

    def det_swipe_down(self):
        # thumb doesn't move much
        # all four fingers move down fast
        if (self.IFT.average_vy > 0.5 and self.IFT.time_acc > self.interval):
            print_log('Swipe Down')
            self.IFT.reset()
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
