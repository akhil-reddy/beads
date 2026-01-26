"""
    An RGB image (or a frame from a video) is processed into a varying length vector matrix
"""
import cv2


def get_stimulus_frame(image):
    # Use vision cell functions
    pass


if __name__ == "__main__":
    # Load vision cells

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
        exit()

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Empty camera frame.")
            break

        get_stimulus_frame(frame)
        # cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
