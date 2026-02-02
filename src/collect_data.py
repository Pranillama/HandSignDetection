#!/usr/bin/env python3
"""Data collection script for capturing hand sign images from webcam."""

import cv2


def main():
    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        if not success:
            break
        cv2.imshow("Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
