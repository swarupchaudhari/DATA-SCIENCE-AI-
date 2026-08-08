import cv2
import mediapipe as mp


class HandGesture:

    def __init__(self):

        # Initialize MediaPipe Hands
        self.mpHands = mp.solutions.hands

        self.hands = self.mpHands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        self.draw = mp.solutions.drawing_utils

    def detect(self, frame):

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = self.hands.process(rgb)

        # Default Gesture
        gesture = "Happy"

        if result.multi_hand_landmarks:

            for hand in result.multi_hand_landmarks:

                # Draw hand landmarks
                self.draw.draw_landmarks(
                    frame,
                    hand,
                    self.mpHands.HAND_CONNECTIONS
                )

                lm = hand.landmark

                fingers = []

                # Thumb
                fingers.append(lm[4].x < lm[3].x)

                # Index
                fingers.append(lm[8].y < lm[6].y)

                # Middle
                fingers.append(lm[12].y < lm[10].y)

                # Ring
                fingers.append(lm[16].y < lm[14].y)

                # Pinky
                fingers.append(lm[20].y < lm[18].y)

                count = fingers.count(True)

                # Gesture Mapping
                if count == 5:
                    gesture = "Happy"

                elif count == 4:
                    gesture = "Emotional"

                elif count == 3:
                    gesture = "Cute"

                elif count == 2:
                    gesture = "Surprise"

                elif count == 1:
                    gesture = "Shock"

                elif count == 0:
                    gesture = "Angry"

                # Display Gesture
                cv2.putText(
                    frame,
                    f"Gesture : {gesture}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )

        return frame, gesture