from instruments.flir import Camera, TriggerType
from PySpin import PySpin
import time

path_to_images = r'C:\Users\ARPA-E\Documents\FLIR TEST\SAFE_GRAB'
# path_to_images = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\CAMERA\LASER_TRIGGER'
acquisition_time = 1.0


def main():
    cam = Camera()
    cam.path_to_images = path_to_images
    cam.image_prefix = 'THROUGHPUT_TEST'
    cam.print_device_info()
    cam.gain = 4.0
    cam.frame_rate = 200
    cam.exposure = 5
    cam.acquisition_time = acquisition_time
    # cam.number_of_images = 5
    # cam.gamma = 0.5
    cam.disable_gamma()
    # cam.trigger_delay = 9
    print(f'Current Gain: {cam.gain}')
    print(f'The exposure read from the camera: {cam.exposure}')
    print(f'The frame rate read from the camera is: {cam.frame_rate} Hz')
    print(f'The number of images to take: {cam.number_of_images}')
    print(f'The acquisition time is: {cam.acquisition_time} s')
    print(f'The trigger delay is: {cam.trigger_delay} us')
    cam.chosen_trigger = TriggerType.SOFTWARE
    cam.acquisition_mode = PySpin.AcquisitionMode_MultiFrame
    # cam.configure_trigger(trigger_type=PySpin.TriggerSelector_FrameStart)
    cam.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)
    cam.acquire_images()
    time.sleep(5.0)
    cam.reset_frame_rate()
    cam.reset_trigger()
    cam.reset_exposure()
    cam.reset_gain()


if __name__ == '__main__':
    main()
