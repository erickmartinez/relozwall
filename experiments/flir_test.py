from instruments.flir import Camera, TriggerType
from PySpin import PySpin
import time

path_to_images = r'C:\Users\ARPA-E\Documents\FLIR TEST'
# path_to_images = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\CAMERA\LCT_graphite_plate_open_iris_050PCT_2022-11-10_3_images'
acquisition_time = 30.0

def main():
    cam = Camera()
    cam.path_to_images = path_to_images
    cam.image_prefix = 'BUFFERS'
    cam.print_device_info()
    cam.gain=0.0
    cam.frame_rate = 200
    cam.exposure = 30E6
    # cam.acquisition_time = acquisition_time
    cam.number_of_images = 5

    print(f'Current Gain: {cam.gain}')
    print(f'The exposure read from the camera: {cam.exposure}')
    print(f'The frame rate read from the camera is: {cam.frame_rate} Hz')
    print(f'The number of images to take: {cam.number_of_images}')
    print(f'The acquisition time is: {cam.acquisition_time} s')
    cam.chosen_trigger = TriggerType.SOFTWARE
    cam.acquisition_mode = PySpin.AcquisitionMode_Continuous
    cam.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)

    cam.acquire_images()
    time.sleep(cam.exposure/1E6)
    cam.reset_frame_rate()
    cam.reset_trigger()
    cam.reset_exposure()
    cam.reset_gain()



if __name__ == '__main__':
    main()
