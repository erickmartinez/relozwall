import pyspin
from pyspin import PySpin
import os
import time


class TriggerType:
    SOFTWARE = 1
    HARDWARE = 2

trigger_type_map = {
    TriggerType.SOFTWARE: 'software', TriggerType.HARDWARE: 'hardware'
}

trigger_start_map = {
    PySpin.TriggerSelector_AcquisitionStart: 'acquisition start',
    PySpin.TriggerSelector_FrameStart: 'frame start',
    PySpin.TriggerSelector_FrameBurstStart: 'frame burst start'
}

acquisition_mode_map = {
    PySpin.AcquisitionMode_Continuous: 'continuous',
    PySpin.AcquisitionMode_MultiFrame: 'multi frame',
    PySpin.AcquisitionMode_SingleFrame: 'single frame'
}

class Camera:
    """
    This class provides methods to control a FLIR camera using the Spinnaker library.
    It assumes that only one camera is connected to the PC.

    Attributes
    ----------
    exposure_time: float
        The exposure time in microseconds
    gain: float
        The gain of the camera
    frame_rate: float
        The acquisition frame rate in Hz
    """
    _exposure_time: float = 100000.0
    _exposure_auto: bool = True
    _gain: float = 5.0
    _gain_auto: bool = True
    _frame_rate: float = 200.0
    _frame_rate_enable: bool = False
    _path_to_images: str = '../data/images'
    _chosen_trigger: TriggerType = TriggerType.SOFTWARE
    _number_of_images: int = 1

    def __init__(self):
        self._system: pyspin.System = pyspin.System.GetInstance()
        self._cam_list: pyspin.CameraList = self._system.GetCameras()
        self._cam: pyspin.Camera = self._cam_list[0]
        try:
            self._cam.Init()
            self._nodemap: pyspin.NodeMap = self._cam.GetNodeMap()
            self._path_to_images = self._path_to_images
        except PySpin.SpinnakerException as ex:
            raise Exception(f'Error: {ex}')

    @property
    def number_of_images(self) -> int:
        return self._number_of_images

    @number_of_images.setter
    def number_of_images(self, number_of_images_to_set: int):
        number_of_images_to_set = int(abs(number_of_images_to_set))
        if number_of_images_to_set > 0:
            self.acquisition_frame_count = number_of_images_to_set

    @property
    def acquisition_time(self) -> float:
        return min(int(self._number_of_images / self._frame_rate), 1)

    @acquisition_time.setter
    def acquisition_time(self, acquisition_time_s):
        self.number_of_images = int(acquisition_time_s * self._frame_rate)

    @property
    def chosen_trigger(self) -> TriggerType:
        return self._chosen_trigger

    @chosen_trigger.setter
    def chosen_trigger(self, new_chosen_trigger):
        if new_chosen_trigger == TriggerType.SOFTWARE or new_chosen_trigger == TriggerType.HARDWARE:
            self._chosen_trigger = new_chosen_trigger
            print(f'Changing trigger type to {trigger_type_map[new_chosen_trigger]}')
        else:
            print(f'The trigger must be an instance of TriggerType. Got {new_chosen_trigger}')

    @property
    def path_to_images(self) -> str:
        return self._path_to_images

    @path_to_images.setter
    def path_to_images(self, new_path: str):
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        self._path_to_images = new_path

    def print_device_info(self):
        print('*** DEVICE INFORMATION ***\n')
        try:
            nodemap_tldevice: pyspin.NodeMap = self._cam.GetTLDeviceNodeMap()
            node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    print('%s: %s' % (node_feature.GetName(), node_feature.ToString() \
                        if PySpin.IsReadable(node_feature) else 'Node not readable'))
            else:
                print('Device control information not available.')
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)

    @property
    def gain(self) -> float:
        try:
            if self._cam.Gain.GetAccessMode() < PySpin.RO:
                print('Unable to read the gain')
                return 0
            gain = float(self._cam.Gain.GetValue())

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return 0
        return gain

    @gain.setter
    def gain(self, gain_value_db):
        self.configure_gain(gain_value_db=gain_value_db)

    @property
    def acquisition_mode(self):
        if self._cam.AcquisitionMode.GetAccessMode() < PySpin.RO:
            print('Unable to access acquisition mode.')
            return 'Error'
        return acquisition_mode_map[self._cam.AcquisitionMode.GetValue()]

    @acquisition_mode.setter
    def acquisition_mode(self, acquisition_mode_to_set):
        if acquisition_mode_to_set not in acquisition_mode_map:
            print(f"Invalid acquisition mode. Given: {acquisition_mode_to_set}.")
            print("Available modes are: %s" % "\n,".join(acquisition_mode_map))
            return False
        if self._cam.AcquisitionMode.GetAccessMode() != PySpin.RW:
            print('Unable to change acquisition mode.')
            return False
        self._cam.AcquisitionMode.SetValue(acquisition_mode_to_set)

    def configure_gain(self, gain_value_db: float):
        try:
            if self._cam.GainAuto.GetAccessMode() != PySpin.RW:
                print('Unable to disable automatic gain. Aborting...')
                return False

            self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            print('Automatic gain disabled...')

            if self._cam.Gain.GetAccessMode() != PySpin.RW:
                print('Unable to set gain. Aborting...')
                return False
            gain_value_db = min(self._cam.Gain.GetMax(), gain_value_db)
            self._cam.Gain.SetValue(gain_value_db)
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            self.reset_gain()
            return False
        self._gain = gain_value_db
        return True

    def reset_gain(self):
        try:
            if self._cam.GainAuto.GetAccessMode() != PySpin.RW:
                print('Unable to disable automatic gain. Aborting...')
                return False
            self._cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)

            print('Automatic gain enabled...')
        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
        return True

    @property
    def frame_rate(self) -> float:
        try:
            if self._cam.AcquisitionFrameRate.GetAccessMode() < PySpin.RO:
                print('Unable to change framerate.')
                return 0
            frame_rate = float(self._cam.AcquisitionFrameRate.GetValue())
        except PySpin.SpinnakerException as ex:
            print(f'Error {ex}')
            return 0
        return frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate_value: float):
        frame_rate_value = abs(frame_rate_value)
        self.configure_frame_rate(frame_rate_value=frame_rate_value)

    def configure_frame_rate(self, frame_rate_value: float):
        try:
            if self._cam.AcquisitionFrameRateEnable.GetAccessMode() != PySpin.RW:
                print('Unable to enable frame acquisition frame rate')
                return False
            self._cam.AcquisitionFrameRateEnable.SetValue(True)
            if self._cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                print('Unable to change framerate')
                self.reset_frame_rate()
                return False
            frame_rate_value = min(self._cam.AcquisitionFrameRate.GetMax(), frame_rate_value)
            self._frame_rate = frame_rate_value
            self._cam.AcquisitionFrameRate.SetValue(frame_rate_value)
            # self._cam.AcquisitionFrameRateEnable.SetValue(True)
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            self.reset_frame_rate()
            return False
        return True

    def reset_frame_rate(self):
        try:
            if self._cam.AcquisitionFrameRateEnable.GetAccessMode() != PySpin.RW:
                print('Unable to change framerate')
                return False
            self._cam.AcquisitionFrameRateEnable.SetValue(False)
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return False
        return True

    @property
    def exposure(self) -> float:
        try:
            if self._cam.ExposureTime.GetAccessMode() < PySpin.RO:
                print('Unable to read exposure time.')
                return 0
            exposure_time = float(self._cam.ExposureTime.GetValue())
        except PySpin.SpinnakerException as ex:
            print(f'Error {ex}')
            return 0
        return exposure_time

    @exposure.setter
    def exposure(self, exposure_time_us: float):
        return self.configure_exposure(exposure_time_us=exposure_time_us)

    def configure_exposure(self, exposure_time_us):
        try:
            if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
                return False

            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            if self._cam.ExposureTime.GetAccessMode() != PySpin.RW:
                print('Unable to set exposure time. Aborting...')
                return False
            exposure_time_us = min(self._cam.ExposureTime.GetMax(), exposure_time_us)
            self._cam.ExposureTime.SetValue(exposure_time_us)
            print('Shutter time set to %s us...\n' % exposure_time_us)
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return False
        return True

    def reset_exposure(self):
        try:
            if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
                return False
            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
            print('Automatic exposure enabled...')
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return False
        return True

    def configure_trigger(self, trigger_type=PySpin.TriggerSelector_AcquisitionStart, trigger_delay_us=9):
        print('*** CONFIGURING TRIGGER ***')

        print('Note that if the application / user software triggers faster than frame time, the trigger may be '
              'dropped / skipped by the camera.')
        print('If several frames are needed per trigger, a more reliable alternative for such case, is to use the '
              'multi-frame mode.\n')
        trigger_delay_us = int(abs(trigger_delay_us))
        if trigger_delay_us == 0:
            trigger_delay_us = 9
        if trigger_type not in trigger_start_map:
            trigger_type = PySpin.TriggerSelector_AcquisitionStart

        if self._chosen_trigger == TriggerType.SOFTWARE:
            print('Software trigger chosen ...')
        elif self._chosen_trigger == TriggerType.HARDWARE:
            print('Hardware trigger chose ...')

        try:
            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source
            # is software or hardware.
            if self._cam.TriggerMode.GetAccessMode() != PySpin.RW:
                print('Unable to disable trigger mode. Aborting...')
                return False
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            print('Trigger mode disabled...')
            if self._cam.TriggerSelector.GetAccessMode() != PySpin.RW:
                print('Unable to get trigger selector. Aborting...')
                return False
            self._cam.TriggerSelector.SetValue(trigger_type)
            print(f'Trigger selector set to {trigger_start_map[trigger_type]}')
            if self._cam.TriggerDelay.GetAccessMode() != PySpin.RW:
                print('Unable to set trigger delay. Aborting...')
            self._cam.TriggerDelay.SetValue(trigger_delay_us)
            if self._cam.TriggerSource.GetAccessMode() != PySpin.RW:
                print('Unable to set trigger source (node retrieval). Aborting...')
                return False
            if self._chosen_trigger == TriggerType.SOFTWARE:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                print('Trigger source set to software...')
            elif self._chosen_trigger == TriggerType.HARDWARE:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                print('Trigger source set to hardware...')
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            print('Trigger mode turned back on...')

        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return False
        return True

    def reset_trigger(self):
        try:
            if self._cam.TriggerMode.GetAccessMode() != PySpin.RW:
                print('Unable to disable trigger mode. Aborting...')
                return False
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            if self._cam.TriggerSelector.GetAccessMode() != PySpin.RW:
                print('Unable to set trigger selector to acquisition start')
            self._cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
        except PySpin.SpinnakerException as ex:
            print(f'Error {ex}')
            return False
        print('Reset trigger')
        return True

    @property
    def acquisition_frame_count(self):
        if self._cam.AcquisitionFrameCount.GetAccessMode() < PySpin.RO:
            print('Error accessing acquisition frame count')
            return 0
        return int(self._cam.AcquisitionFrameCount.GetValue())

    @acquisition_frame_count.setter
    def acquisition_frame_count(self, frame_count: int):
        frame_count = max(2, int(abs(frame_count)))
        if self._cam.AcquisitionFrameCount.GetAccessMode() != PySpin.RW:
            print('Unable to change acquisition frame count')
            return False
        frame_count = min(self._cam.AcquisitionFrameCount.GetMax(), frame_count)
        self._cam.AcquisitionFrameCount.SetValue(frame_count)
        self._number_of_images = frame_count
        return True

    def start_acquisition(self):
        try:
            self._cam.BeginAcquisition()
            print('Acquiring images...')
        except PySpin.SpinnakerException as ex:
            print(f'Error {ex}')
            return False

    def acquire_images(self):
        try:
            # self.acquisition_mode = PySpin.AcquisitionMode_Continuous
            # print('Acquisition mode set to multi frame...')
            #  Begin acquiring images
            self._cam.BeginAcquisition()
            print('Acquiring images...')
            device_serial_number = self.device_serial_number
            print('Device serial number retrieved as %s...' % device_serial_number)
            # Retrieve, convert, and save images

            # Create ImageProcessor instance for post processing images
            processor = PySpin.ImageProcessor()

            # Set default image processor color processing method
            #
            # *** NOTES ***
            # By default, if no specific color processing algorithm is set, the image
            # processor will default to NEAREST_NEIGHBOR method.
            processor.SetColorProcessing(PySpin.HQ_LINEAR)

            # Get the value of exposure time to set an appropriate timeout for GetNextImage
            exposure = self.exposure
            if exposure == 0:
                print('Error retrieving the exposure time. Aborting image acquisition...')
                return False
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency
            # with the unit being used in GetNextImage
            timeout = (int)(1000.0 / self.frame_rate + 10)
            # timeout = (int)(self._cam.ExposureTime.GetValue() / 1000 + 10)
            print(f'Acquisition timeout: {timeout}')
            self.execute_trigger()
            previous_seconds = 0
            elapsed_time = 0
            i = 0
            for i in range(self._number_of_images):
                # current_seconds = time.time()
                # if (current_seconds - previous_seconds) >= 1.0:
                try:
                    # self.grab_next_image_by_trigger()
                    image_result = self._cam.GetNextImage(timeout)
                    if image_result.IsIncomplete():
                        print('Image incomplete with image status %d...' % image_result.GetImageStatus())

                    else:
                        # Print image information
                        width = image_result.GetWidth()
                        height = image_result.GetHeight()
                        print('Grabbed Image %d/%d, width = %d, height = %d' % (i, self._number_of_images, width, height))

                        # Convert image to Mono8
                        image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                        # Create a unique filename
                        filename = 'Experiment-%s-%d.jpg' % (device_serial_number, i)
                        full_filename = os.path.join(self._path_to_images, filename)

                        # Save image
                        image_converted.Save(full_filename)

                        print('Image saved at %s' % full_filename)

                        # Release image
                        image_result.Release()
                        # print(f'Released image {i}')
                        i += 1
                except PySpin.SpinnakerException as ex:
                    print('Error: %s' % ex)
                    return False
                    # self.execute_trigger()
                # previous_seconds = current_seconds
            self._cam.EndAcquisition()
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return False
        # self.acquisition_mode = PySpin.AcquisitionMode_Continuous
        # print('Acquisition mode set back to continuous...')
        return True

    def grab_next_image_by_trigger(self):
        if self._chosen_trigger == TriggerType.SOFTWARE:
            # if self._cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
            #     raise PySpin.SpinnakerException('Unable to execute trigger. Aborting...')
            self._cam.TriggerSoftware.Execute()
            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
        elif self._chosen_trigger == TriggerType.HARDWARE:
            print('Use the hardware to trigger image acquisition.')

    def execute_trigger(self):
        if self._chosen_trigger == TriggerType.SOFTWARE:
            # if self._cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
            #     raise PySpin.SpinnakerException('Unable to execute trigger. Aborting...')
            self._cam.TriggerSoftware.Execute()
            # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
        elif self._chosen_trigger == TriggerType.HARDWARE:
            print('Use the hardware to trigger image acquisition.')

    @property
    def device_serial_number(self) -> str:
        try:
            if self._cam.DeviceSerialNumber.GetAccessMode() < PySpin.RO:
                print('Unable to read the serial number')
                return ''
            serial_number = str(self._cam.DeviceSerialNumber.GetValue())
        except PySpin.SpinnakerException as ex:
            print(f'Error: {ex}')
            return ''
        return serial_number

    def __del__(self):
        self._cam.DeInit()
        del self._cam
        self._cam_list.Clear()
        self._system.ReleaseInstance()
