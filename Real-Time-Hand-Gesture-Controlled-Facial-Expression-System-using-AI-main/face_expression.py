import cv2
import os


class FaceExpression:

    def __init__(self):

        # Dictionary of emotion images
     self.images = {
    "Happy": "images/Happy.jpg",
    "Angry": "images/Angry.jpg",
    "Surprise": "images/Surprise.jpg",
    "Shock": "images/Shock.jpg",
    "Cute": "images/Cute.jpg",
    "Emotional": "images/Emotional.jpg"
}

    def show_expression(self, frame, emotion):

        # Get image path
        image_path = self.images.get(emotion)

        # If emotion not found
        if image_path is None:
            print(f"❌ Emotion '{emotion}' not found.")
            return frame

        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return frame

        # Read image
        img = cv2.imread(image_path)

        if img is None:
            print(f"❌ Unable to load image: {image_path}")
            return frame

        # Resize image
        img = cv2.resize(img, (300, 300))

        # Put image on webcam frame
        frame[30:330, 20:320] = img

        # Display emotion text
        cv2.putText(
            frame,
            f"Expression : {emotion}",
            (20, 370),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return frame