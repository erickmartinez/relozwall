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

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv2.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = False
    params.minArea = 2

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Set Convexity filtering parameters
    params.filterByConvexity = False
    params.minConvexity = 0.2

    # Set inertia filtering parameters
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(imgb)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv2.drawKeypoints(cimg, keypoints, blank, (0, 0, 255),
                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    text = "Number of Circular Blobs: " + str(len(keypoints))
    cv2.putText(blobs, text, (20, 550),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)

    # Show blobs
    cv2.imshow("Filtering Circular Blobs Only", blobs)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()