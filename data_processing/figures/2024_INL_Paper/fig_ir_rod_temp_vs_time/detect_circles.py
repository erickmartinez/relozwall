import cv2
import numpy as np
import os


base_dir = r'/Users/erickmartinez/Library/CloudStorage/OneDrive-Personal/Documents/ucsd/Postdoc/research/data/firing_tests/SS_TUBE/GC/LCT_R4N137_ROW513_100PCT_2024-01-24_1_images'

file = 'R4N137_ROW513_IMG-12-17075589111320.tiff'

def main():
    global base_dir, file
    full_file = os.path.join(base_dir, file)
    img = cv2.imread(full_file, cv2.IMREAD_GRAYSCALE)
    imgb = cv2.medianBlur(img, 5)
    cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Apply Hough transform on the blurred image.
    detected_circles = cv2.HoughCircles(img,
                                        cv2.HOUGH_GRADIENT, 1, 8, param1=20,
                                        param2=16, minRadius=0, maxRadius=14)

    # Draw circles that are detected.
    if detected_circles is not None:
        # Convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))

        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]

            # Draw the circumference of the circle.
            cv2.circle(cimg, (a, b), r, (0, 255, 0), 1)

            # Draw a small circle (of radius 1) to show the center.
            # cv2.circle(cimg, (a, b), 1, (0, 0, 255), 1)
            cv2.imshow("Detected Circle", cimg)

    cv2.waitKey(0)

if __name__ == '__main__':
    main()