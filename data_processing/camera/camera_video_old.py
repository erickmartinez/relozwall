import numpy as np
import os
import cv2

data_path = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\SAMPLES\BINDER_CONTENT_SCAN\LCT_R4N14_ROW157_100PCT_2022-12-06_1_images'
file_tag = 'LCT_R4N14_ROW157_100PCT_2022-12-06_1'
frame_rate = 64.30


def get_files(base_dir: str, tag: str):
    files = []
    for f in os.listdir(base_dir):
        if f.startswith(tag) and f.endswith('.jpg'):
            files.append(f)
    return files


def main():
    list_of_files = get_files(base_dir=data_path, tag=file_tag)
    frameSize = (1440, 1080)

    out = cv2.VideoWriter(
        # os.path.join(base_path, f'{file_tag}.avi'), cv2.VideoWriter_fourcc(*'DIVX'), 5, frameSize
        os.path.join(data_path, f'{file_tag}.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), frame_rate / 10, frameSize
    )

    for i, f in enumerate(list_of_files):
        t = (i + 1) / frame_rate + 9E-6
        txt = f'{t:>4.3f} s'
        color = (0, 0, 255, 255) if t <= 0.5 else (255, 255, 255, 255)
        img = cv2.imread(os.path.join(data_path, f))
        position = (10, 75)
        cv2.putText(
            img,  # numpy array on which text is written
            txt,  # text
            position,  # position at which writing has to start
            cv2.FONT_HERSHEY_SIMPLEX,  # font family
            3,  # font size
            color,  # font color
            3)  # font stroke
        out.write(img)

    out.release()


if __name__ == '__main__':
    main()
