import cv2, csv
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np
def landmarks_to_vector(landmarks):
    if(landmarks):
        landmark_flat_vec = np.array([[landmarks.landmark[i].x, landmarks.landmark[i].y, landmarks.landmark[i].z] 
                              for i in range(21)]).flatten()
    else:
        landmark_flat_vec = np.empty(63)
        landmark_flat_vec[:] = np.nan 
    return landmark_flat_vec
def stream(save_trace = False, fname = "./data/default.csv"):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if save_trace:
        with open(fname, 'w', newline='') as file:
            writer = csv.writer(file)
            header = ['x_{}'.format(i) for i in range(63)] 
            writer.writerow(header)

            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        label = handedness.classification[0].label
                        if label == 'Right':
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                            vector = landmarks_to_vector(hand_landmarks)
                            writer.writerow(vector)

                cv2.imshow('MediaPipe Hands', image)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
    else:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    if handedness.classification[0].label == 'Right':
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break


    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    stream(save_trace=True)