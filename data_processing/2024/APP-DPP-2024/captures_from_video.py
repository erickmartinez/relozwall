import cv2 as cv
import os
import time

# Read the video from specified path
cam = cv.VideoCapture(r'./LCT_AMB2R006_ROW588_100PCT_2024-06-12_1_cropped_movie.mp4')

try:

    # creating a folder named data
    if not os.path.exists('video_captures'):
        os.makedirs('video_captures')

    # if not created then raise error
except OSError:
    print('Error: Creating directory of data')

# frame
currentframe = 0

while (True):
    time.sleep(5) # take schreenshot every 5 seconds
    # reading from frame
    ret, frame = cam.read()

    if ret:
        # if video is still left continue creating images
        name = './video_captures/frame' + str(currentframe) + '.png'
        print('Creating...' + name)

        # writing the extracted images
        cv.imwrite(name, frame)

        # increasing counter so that it will
        # show how many frames are created
        currentframe += 1
    else:
        break

# Release all space and windows once done
cam.release()
cv.destroyAllWindows()