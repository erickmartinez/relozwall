import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy.linalg as la

data_path = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\perspective_matrix'
img_file = 'scale_20230324_90deg.bmp'

# To open matplotlib in interactive mode
# matplotlib qt5

def rotation_matrix(angle):
    pass


def main():
    # Load the image
    img = cv2.imread(os.path.join(data_path, img_file))

    # Create a copy of the image
    img_copy = np.copy(img)

    # Convert to RGB so as to display via matplotlib
    # Using Matplotlib we can easily find the coordinates
    # of the 4 points that is essential for finding the
    # transformation matrix
    img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB)

    # All points are in format [cols, rows]
    pt_A = [31, 121]
    pt_B = [31, 480]
    pt_C = [1043, 475]
    pt_D = [1036, 3]

    width_AD = la.norm([pt_A[0] - pt_D[0], pt_A[1] - pt_D[1]])  #np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = la.norm([pt_B[0] - pt_C[0], pt_B[1] - pt_C[1]])# np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))

    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    output_pts = np.float32([[0, 0],
                             [0, maxHeight - 1],
                             [maxWidth - 1, maxHeight - 1],
                             [maxWidth - 1, 0]])
    # output_pts = np.float32([[0, 0],
    #                          [0, maxWidth - 1],
    #                          [maxWidth - 1, maxHeight - 1],
    #                          [0, maxHeight - 1]])

    # Compute the perspective transform M
    M = cv2.getPerspectiveTransform(input_pts, output_pts)
    out = cv2.warpPerspective(img_copy, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)

    fig, axes = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    fig.set_size_inches(4., 6)
    axes[0].imshow(img_copy)
    axes[1].imshow(out)


    # fig2, axes2 = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    # fig2.set_size_inches(4., 6)
    # out2 =

    plt.show()

if __name__ == '__main__':
    main()