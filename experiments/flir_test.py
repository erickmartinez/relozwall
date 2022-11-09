from instruments.flir import Camera, TriggerType
import pyspin.PySpin as PySpin

path_to_images = r'C:\Users\erick\OneDrive\Documents\ucsd\Postdoc\research\thermal camera\test'
acquisition_time = 2.0

def main():
    cam = Camera()
    cam.path_to_images = path_to_images
    cam.print_device_info()
    cam.gain=5.0
    cam.frame_rate = 100
    cam.exposure = 10000
    cam.acquisition_time = acquisition_time
    print(f'Current Gain: {cam.gain}')
    print(f'The exposure read from the camera: {cam.exposure}')
    print(f'The frame rate read from the camera is: {cam.frame_rate} Hz')
    print(f'The number of images to take: {cam.number_of_images}')
    print(f'The acquisition time is: {cam.acquisition_time} s')
    cam.chosen_trigger = TriggerType.SOFTWARE
    cam.acquisition_mode = PySpin.AcquisitionMode_Continuous
    cam.configure_trigger(trigger_type=PySpin.TriggerSelector_AcquisitionStart)

    cam.acquire_images()
    cam.reset_frame_rate()
    cam.reset_trigger()
    cam.reset_exposure()
    cam.reset_gain()


if __name__ == '__main__':
    main()
