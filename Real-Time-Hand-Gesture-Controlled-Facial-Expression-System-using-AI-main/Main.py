import cv2

from Gesture_Detector import HandGesture
from face_expression import FaceExpression


def main():

    # Open Webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("❌ Cannot open webcam.")
        return

    gesture = HandGesture()
    face = FaceExpression()

    while True:

        success, frame = camera.read()

        if not success:
            print("❌ Failed to capture frame.")
            break

        # Mirror image
        frame = cv2.flip(frame, 1)

        # Detect gesture
        frame, emotion = gesture.detect(frame)

        # Display expression image
        frame = face.show_expression(frame, emotion)

        # Show output
        cv2.imshow(
            "Real-Time Hand Gesture Controlled Facial Expression System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        # Press ESC or Q to exit
        if key == 27 or key == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()