import cv2


def draw_points(frame, landmarks):

    for name, point in landmarks.items():

        cv2.circle(frame, point, 8, (255, 0, 0), -1)

        cv2.putText(
            frame,
            name,
            (point[0] + 10, point[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )