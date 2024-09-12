import sys
import os
from instruments.flir import Camera, TriggerType
from PySpin import PySpin
import time

path_to_images = r'C:\Users\ARPA-E\Documents\FLIR TEST\SAFE_GRAB'
# path_to_images = r'G:\Shared drives\ARPA-E Project\Lab\Data\Laser Tests\CAMERA\LASER_TRIGGER'
acquisition_time = 2.0


def main():
    files_in_dir = os.listdir(path_to_images)
    for f in files_in_dir:
        fp = os.path.join(path_to_images, f)
        os.remove(fp)
    cam = Camera()
    cam.path_to_images = path_to_images
    cam.image_prefix = 'THROUGHPUT_TEST'
    cam.print_device_info()
    cam.gain = 4.0
    cam.frame_rate = 200
    cam.exposure = 500
    cam.acquisition_time = acquisition_time
    # cam.number_of_images = 5
    # cam.gamma = 0.5
    cam.disable_gamma()
    # cam.trigger_delay = 9
    # Retrieve singleton reference to system object
    system = PySpin.System.GetInstance()

    # Get current library version
    version = system.GetLibraryVersion()
    print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))
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
    time.sleep(1.0)
    cam.reset_frame_rate()
    cam.reset_trigger()
    cam.reset_exposure()
    cam.reset_gain()
    try:
        cam.shutdown()
    except PySpin.SpinnakerException as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    main()
