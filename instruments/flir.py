import time

import PySpin
import os
import logging
from exif import Image as ImageInfo

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

def modify_img_metadata(path_to_img: str, meta_dict: dict):
    try:
        with open(path_to_img, 'rb') as img_file:
            img = ImageInfo(img_file)
        for key in meta_dict:
            val = meta_dict[key]
            img.key = val
        with open(path_to_img, 'wb') as new_img_file:
            new_img_file.write(img.get_file())
    except Exception as ex:
        print(ex)


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
    _log: logging.Logger = None
    _image_prefix: str = None
    _print_info: bool = False
    _fast_timeout: bool = False
    _cam: PySpin.Camera = None
    __busy: bool = False
    debug: bool = False

    def __init__(self):
        self._system: PySpin.System = PySpin.System.GetInstance()
        self._cam_list: PySpin.CameraList = self._system.GetCameras()
        self._cam: PySpin.Camera = self._cam_list[0]
        try:
            self._cam.Init()
            self._nodemap: PySpin.NodeMap = self._cam.GetNodeMap()
            self._path_to_images = self._path_to_images
            # self._cam.DeviceMaxThroughput.SetValue(811057600)  # 311057600 <<<< Not writable
            self._cam.DeviceLinkThroughputLimit.SetValue(500000000)
            self._cam.ChunkModeActive.SetValue(True)
            self._cam.ChunkSelector.SetValue(PySpin.ChunkSelector_FrameID)
            self._cam.ChunkEnable.SetValue(True)
            self._cam.ChunkSelector.SetValue(PySpin.ChunkSelector_Timestamp)
            self._cam.ChunkEnable.SetValue(True)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}')
            raise Exception(f'Error: {ex}')
        self.disable_gamma()

    @property
    def busy(self) -> bool:
        return self.__busy

    @property
    def number_of_images(self) -> int:
        if self._cam.AcquisitionMode.GetAccessMode() >= PySpin.RO:
            acquisition_mode = self._cam.AcquisitionMode.GetValue
            if acquisition_mode == PySpin.AcquisitionMode_SingleFrame:
                return 1
        return self.acquisition_frame_count

    @number_of_images.setter
    def number_of_images(self, number_of_images_to_set: int):
        number_of_images_to_set = int(abs(number_of_images_to_set))
        if number_of_images_to_set > 0:
            self.acquisition_frame_count = number_of_images_to_set

    @property
    def acquisition_time(self) -> float:
        return self.acquisition_frame_count / self._frame_rate

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
            self.log(f'Changing trigger type to {trigger_type_map[new_chosen_trigger]}')
        else:
            self.log(f'The trigger must be an instance of TriggerType. Got {new_chosen_trigger}')

    @property
    def path_to_images(self) -> str:
        return self._path_to_images

    @path_to_images.setter
    def path_to_images(self, new_path: str):
        if not os.path.exists(new_path):
            os.makedirs(new_path)
        self._path_to_images = new_path

    @property
    def image_prefix(self) -> str:
        return self._image_prefix

    @image_prefix.setter
    def image_prefix(self, prefix):
        self._image_prefix = prefix

    @property
    def fast_timeout(self) -> bool:
        return self._fast_timeout

    @fast_timeout.setter
    def fast_timeout(self, value: bool):
        self._fast_timeout = bool(value)

    @property
    def trigger_delay(self) -> int:
        return int(self._cam.TriggerDelay.GetValue())

    @trigger_delay.setter
    def trigger_delay(self, delay_to_set: int):
        delay_to_set = int(abs(delay_to_set))
        if self._cam.TriggerDelay.GetAccessMode() != PySpin.RW:
            self.log('Unable to access the trigger delay in write mode', logging.WARNING)
        delay_to_set = max(9, delay_to_set)
        delay_to_set = min(self._cam.TriggerDelay.GetMax(), delay_to_set)
        self._cam.TriggerDelay.SetValue(delay_to_set)

    def print_device_info(self):
        self.log('*** DEVICE INFORMATION ***\n')
        try:
            nodemap_tldevice: PySpin.NodeMap = self._cam.GetTLDeviceNodeMap()
            node_device_information = PySpin.CCategoryPtr(nodemap_tldevice.GetNode('DeviceInformation'))
            if PySpin.IsAvailable(node_device_information) and PySpin.IsReadable(node_device_information):
                features = node_device_information.GetFeatures()
                for feature in features:
                    node_feature = PySpin.CValuePtr(feature)
                    self.log('%s: %s' % (node_feature.GetName(), node_feature.ToString() \
                        if PySpin.IsReadable(node_feature) else 'Node not readable'))
            else:
                self.log('Device control information not available.')
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)

    @property
    def gain(self) -> float:
        try:
            if self._cam.Gain.GetAccessMode() < PySpin.RO:
                self.log('Unable to read the gain', logging.ERROR)
                return 0
            gain = float(self._cam.Gain.GetValue())

        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return 0
        return gain

    @gain.setter
    def gain(self, gain_value_db):
        gain_value_db = abs(float(gain_value_db))
        self.configure_gain(gain_value_db=gain_value_db)

    @property
    def acquisition_mode(self):
        if self._cam.AcquisitionMode.GetAccessMode() < PySpin.RO:
            self.log('Unable to access acquisition mode.', logging.ERROR)
            return 'Error'
        return acquisition_mode_map[self._cam.AcquisitionMode.GetValue()]

    @acquisition_mode.setter
    def acquisition_mode(self, acquisition_mode_to_set):
        if acquisition_mode_to_set not in acquisition_mode_map:
            self.log(f"Invalid acquisition mode. Given: {acquisition_mode_to_set}.", logging.ERROR)
            self.log("Available modes are: %s" % "\n,".join(acquisition_mode_map))
            return False
        if self._cam.AcquisitionMode.GetAccessMode() != PySpin.RW:
            self.log('Unable to change acquisition mode.', logging.ERROR)
            return False
        self._cam.AcquisitionMode.SetValue(acquisition_mode_to_set)

    def configure_gain(self, gain_value_db: float):
        try:
            if self._cam.GainAuto.GetAccessMode() != PySpin.RW:
                self.log('Unable to disable automatic gain. Aborting...', logging.ERROR)
                return False

            self._cam.GainAuto.SetValue(PySpin.GainAuto_Off)
            self.log('Automatic gain disabled...')

            if self._cam.Gain.GetAccessMode() != PySpin.RW:
                self.log('Unable to set gain. Aborting...', logging.ERROR)
                return False
            gain_value_db = min(self._cam.Gain.GetMax(), gain_value_db)
            self._cam.Gain.SetValue(gain_value_db)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            self.reset_gain()
            return False
        self._gain = gain_value_db
        return True

    def reset_gain(self):
        try:
            if self._cam.GainAuto.GetAccessMode() != PySpin.RW:
                self.log('Unable to disable automatic gain. Aborting...', logging.ERROR)
                return False
            self._cam.GainAuto.SetValue(PySpin.GainAuto_Continuous)

            self.log('Automatic gain enabled...')
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    @property
    def frame_rate(self) -> float:
        try:
            if self._cam.AcquisitionFrameRate.GetAccessMode() < PySpin.RO:
                self.log('Unable to change framerate.', logging.ERROR)
                return 0
            frame_rate = float(self._cam.AcquisitionFrameRate.GetValue())
        except PySpin.SpinnakerException as ex:
            self.log(f'Error {ex}')
            return 0
        return frame_rate

    @frame_rate.setter
    def frame_rate(self, frame_rate_value: float):
        frame_rate_value = abs(float(frame_rate_value))
        self.configure_frame_rate(frame_rate_value=frame_rate_value)

    def set_buffers(self, num_buffers):
        # Retrieve Stream Parameters device nodemap
        s_node_map = self._cam.GetTLStreamNodeMap()
        # Retrieve Buffer Handling Mode Information
        handling_mode = PySpin.CEnumerationPtr(s_node_map.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsAvailable(handling_mode) or not PySpin.IsWritable(handling_mode):
            self.log('Unable to set Buffer Handling mode (node retrieval). Aborting...', level=logging.WARNING)
            return False

        handling_mode_entry = PySpin.CEnumEntryPtr(handling_mode.GetCurrentEntry())
        if not PySpin.IsAvailable(handling_mode_entry) or not PySpin.IsReadable(handling_mode_entry):
            self.log('Unable to set Buffer Handling mode (Entry retrieval). Aborting...', level=logging.WARNING)
            return False

        # Retrieve and modify Stream Buffer Count
        buffer_count = PySpin.CIntegerPtr(s_node_map.GetNode('StreamBufferCountManual'))
        if not PySpin.IsAvailable(buffer_count) or not PySpin.IsWritable(buffer_count):
            self.log('Unable to set Buffer Count (Integer node retrieval). Aborting...', level=logging.WARNING)
            return False

        # Display Buffer Info
        print('Default Buffer Handling Mode: %s' % handling_mode_entry.GetDisplayName())
        print('Default Buffer Count: %d' % buffer_count.GetValue())
        print('Maximum Buffer Count: %d' % buffer_count.GetMax())

        num_buffers_to_set = min(buffer_count.GetMax(), num_buffers)
        buffer_count.SetValue(buffer_count.GetMax())

        self.log('Buffer count now set to: %d' % buffer_count.GetValue())

        handling_mode_entry = handling_mode.GetEntryByName('OldestFirst')
        handling_mode.SetIntValue(handling_mode_entry.GetValue())
        print('Buffer Handling Mode has been set to %s' % handling_mode_entry.GetDisplayName())

    def configure_frame_rate(self, frame_rate_value: float):
        frame_rate_value = abs(float(frame_rate_value))
        try:
            if self._cam.AcquisitionFrameRateEnable.GetAccessMode() != PySpin.RW:
                self.log('Unable to enable frame acquisition frame rate')
                return False
            self._cam.AcquisitionFrameRateEnable.SetValue(True)
            if self._cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                self.log('Unable to change framerate', logging.ERROR)
                self.reset_frame_rate()
                return False
            frame_rate_value = min(self._cam.AcquisitionFrameRate.GetMax(), frame_rate_value)
            self._frame_rate = frame_rate_value
            self._cam.AcquisitionFrameRate.SetValue(frame_rate_value)
            self.log(f"Set the frame rate to {frame_rate_value:.2f} Hz")
            # self._cam.AcquisitionFrameRateEnable.SetValue(True)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            self.reset_frame_rate()
            return False
        return True

    def reset_frame_rate(self):
        try:
            if self._cam.AcquisitionFrameRateEnable.GetAccessMode() != PySpin.RW:
                self.log('Unable to change framerate', logging.WARNING)
                return False
            self._cam.AcquisitionFrameRateEnable.SetValue(False)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    @property
    def exposure(self) -> float:
        try:
            if self._cam.ExposureTime.GetAccessMode() < PySpin.RO:
                self.log('Unable to read exposure time.', logging.WARNING)
                return 0
            exposure_time = float(self._cam.ExposureTime.GetValue())
        except PySpin.SpinnakerException as ex:
            self.log(f'Error {ex}')
            return 0
        return exposure_time

    @exposure.setter
    def exposure(self, exposure_time_us: float):
        return self.configure_exposure(exposure_time_us=exposure_time_us)

    def configure_exposure(self, exposure_time_us):
        try:
            if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                self.log('Unable to enable automatic exposure (node retrieval). Non-fatal error...', logging.WARNING)
                return False

            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            if self._cam.ExposureTime.GetAccessMode() != PySpin.RW:
                self.log('Unable to set exposure time. Aborting...', logging.ERROR)
                return False
            exposure_time_us = min(self._cam.ExposureTime.GetMax(), exposure_time_us)
            self._cam.ExposureTime.SetValue(exposure_time_us)
            self.log('Shutter time set to %s us...\n' % exposure_time_us)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    def reset_exposure(self):
        try:
            if self._cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                self.log('Unable to enable automatic exposure (node retrieval). Non-fatal error...', logging.WARNING)
                return False
            self._cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)
            self.log('Automatic exposure enabled...')
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    def configure_trigger(self, trigger_type=PySpin.TriggerSelector_AcquisitionStart, trigger_delay_us=9):
        self.log('*** CONFIGURING TRIGGER ***')

        self.log('Note that if the application / user software triggers faster than frame time, the trigger may be '
                 'dropped / skipped by the camera.')
        self.log('If several frames are needed per trigger, a more reliable alternative for such case, is to use the '
                 'multi-frame mode.\n')
        trigger_delay_us = int(abs(trigger_delay_us))
        if trigger_delay_us == 0:
            trigger_delay_us = 9
        if trigger_type not in trigger_start_map:
            trigger_type = PySpin.TriggerSelector_AcquisitionStart

        if self._chosen_trigger == TriggerType.SOFTWARE:
            self.log('Software trigger chosen ...', logging.INFO)
        elif self._chosen_trigger == TriggerType.HARDWARE:
            self.log('Hardware trigger chose ...', logging.INFO)

        try:
            # Ensure trigger mode off
            # The trigger must be disabled in order to configure whether the source
            # is software or hardware.
            if self._cam.TriggerMode.GetAccessMode() != PySpin.RW:
                self.log('Unable to disable trigger mode. Aborting...', logging.ERROR)
                return False
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            self.log('Trigger mode disabled...')
            if self._cam.TriggerSelector.GetAccessMode() != PySpin.RW:
                self.log('Unable to get trigger selector. Aborting...', logging.ERROR)
                return False
            self._cam.TriggerSelector.SetValue(trigger_type)
            self.log(f'Trigger selector set to {trigger_start_map[trigger_type]}')
            if self._cam.TriggerDelay.GetAccessMode() != PySpin.RW:
                self.log('Unable to set trigger delay. Aborting...', logging.ERROR)
            self._cam.TriggerDelay.SetValue(trigger_delay_us)
            if self._cam.TriggerSource.GetAccessMode() != PySpin.RW:
                self.log('Unable to set trigger source (node retrieval). Aborting...', logging.ERROR)
                return False
            if self._chosen_trigger == TriggerType.SOFTWARE:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Software)
                self.log('Trigger source set to software...')
            elif self._chosen_trigger == TriggerType.HARDWARE:
                self._cam.TriggerSource.SetValue(PySpin.TriggerSource_Line0)
                self.log("Changing trigger source to Line0")
                if self._cam.TriggerActivation.GetAccessMode() != PySpin.RW:
                    self.log("Couldn't change trigger activation to Rising Edge", logging.ERROR)
                self._cam.TriggerActivation.SetValue(PySpin.TriggerActivation_RisingEdge)
                self.log("Changing trigger activation to RisingEdge")
                self.log('Trigger source set to hardware...')
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_On)
            self.log('Trigger mode turned back on...')

        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    def reset_trigger(self):
        try:
            if self._cam.TriggerMode.GetAccessMode() != PySpin.RW:
                self.log('Unable to disable trigger mode. Aborting...', logging.ERROR)
                return False
            self._cam.TriggerMode.SetValue(PySpin.TriggerMode_Off)
            if self._cam.TriggerSelector.GetAccessMode() != PySpin.RW:
                self.log('Unable to set trigger selector to acquisition start', logging.ERROR)
            self._cam.TriggerSelector.SetValue(PySpin.TriggerSelector_AcquisitionStart)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        self.log('Reset trigger')
        return True

    @property
    def gamma(self) -> float:
        if self._cam.Gamma.GetAccessMode() < PySpin.RO:
            self.log('Unable to access gamma value. Aborting...', logging.ERROR)
            return
        return float(self._cam.Gamma.GetValue())

    @gamma.setter
    def gamma(self, new_value: float):
        new_value = abs(float(new_value))
        try:
            self.enable_gamma()
            if self._cam.Gamma.GetAccessMode() != PySpin.RW:
                self.log('Unable to change gamma. Aborting...', logging.ERROR)
                self.log(f'Gamma access mode: {self._cam.Gamma.GetAccessMode()}')
                self.log(f'Requested access mode: {PySpin.RW}')
                return
            new_value = max(self._cam.Gamma.GetMin(), new_value)
            new_value = min(self._cam.Gamma.GetMax(), new_value)
            self._cam.Gamma.SetValue(new_value)
            self.log(f'Set the value of gamma to {new_value}')
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    def disable_gamma(self):
        if self._cam.GammaEnable.GetAccessMode() != PySpin.RW:
            self.log('Unable to disable gamma. Aborting...', logging.ERROR)
            return False
        self._cam.GammaEnable.SetValue(False)
        self.log('Disabled gammma')
        return True

    def enable_gamma(self):
        if self._cam.GammaEnable.GetAccessMode() != PySpin.RW:
            self.log('Unable to enable gamma. Aborting...', logging.ERROR)
            return False
        self._cam.GammaEnable.SetValue(True)
        self.log('Enable gammma')
        return True

    @property
    def acquisition_frame_count(self):
        if self._cam.AcquisitionFrameCount.GetAccessMode() < PySpin.RO:
            self.log('Error accessing acquisition frame count', logging.ERROR)
            return 0
        return int(self._cam.AcquisitionFrameCount.GetValue())

    @acquisition_frame_count.setter
    def acquisition_frame_count(self, frame_count: int):
        frame_count = max(2, int(abs(frame_count)))
        try:
            if self._cam.AcquisitionFrameCount.GetAccessMode() != PySpin.RW:
                self.log('Unable to change acquisition frame count', logging.ERROR)
                return False
            frame_count = min(self._cam.AcquisitionFrameCount.GetMax(), frame_count)
            self._cam.AcquisitionFrameCount.SetValue(frame_count)
            if self._cam.AcquisitionBurstFrameCount.GetAccessMode() != PySpin.RW:
                self.log('Unable to change acquisition burst frame count', logging.ERROR)
            burst_frame_count = min(self._cam.AcquisitionBurstFrameCount.GetMax(), frame_count)
            self._cam.AcquisitionBurstFrameCount.SetValue(burst_frame_count)
            self._number_of_images = frame_count
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return False
        return True

    def acquire_images(self):
        try:
            # self.acquisition_mode = PySpin.AcquisitionMode_Continuous
            # self.log('Acquisition mode set to multi frame...')
            # MAX 1440
            self.set_buffers(45)

            # self._cam.Width.SetValue(self._cam.WidthMax.GetValue())
            #  Begin acquiring images
            self._cam.BeginAcquisition()
            self.log('Acquiring images...')
            device_serial_number = self.device_serial_number
            self.log('Device serial number retrieved as %s..' % device_serial_number)
            if self._image_prefix is None:
                image_prefix = 'Experiment_%s' % device_serial_number
            else:
                image_prefix = self._image_prefix
            # Retrieve, convert, and save images

            # Create ImageProcessor instance for post processing images
            processor = PySpin.ImageProcessor()

            # Set default image processor color processing method
            #
            # *** NOTES ***
            # By default, if no specific color processing algorithm is set, the image
            # processor will default to NEAREST_NEIGHBOR method.
            # processor.SetColorProcessing(PySpin.HQ_LINEAR)

            # Get the value of exposure time to set an appropriate timeout for GetNextImage
            exposure = self.exposure
            if exposure == 0:
                self.log('Error retrieving the exposure time. Aborting image acquisition...', logging.ERROR)
                return False
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency
            # with the unit being used in GetNextImage
            fast_timeout = (int) (1000.0 / self.frame_rate + 50 + self.exposure/1000)
            # timeout = (int)(self._cam.ExposureTime.GetValue() / 1000 + 10)
            self.execute_trigger()
            previous_seconds = 0
            elapsed_time = 0
            i = 0
            self.__busy = True
            if self.acquisition_mode == 'single frame':
                self.log('Acquisition mode: single frame')
                try:
                    if self._cam.ExposureTime.GetAccessMode() == PySpin.RW or self._cam.ExposureTime.GetAccessMode() == PySpin.RO:
                        # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit being used in GetNextImage
                        timeout = (int)(self._cam.ExposureTime.GetValue() / 1000 + 1000 + self.trigger_delay/1000)
                        self.log(f'Acquisition timeout: {timeout}')
                    image_result = self._cam.GetNextImage(timeout)

                    if image_result.IsIncomplete():
                        self.log('Image incomplete with image status %d...' % image_result.GetImageStatus(),
                                 logging.WARNING)

                    else:
                        # Print image information
                        if self.debug:
                            width = image_result.GetWidth()
                            height = image_result.GetHeight()
                            self.log(
                                'Grabbed image 1/1, width = %d, height = %d' % (width, height))

                        # Convert image to Mono8
                        image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                        # Create a unique filename
                        filename = '%s-%d.jpg' % (image_prefix, i+1)
                        full_filename = os.path.join(self._path_to_images, filename)

                        # Save image
                        image_converted.Save(full_filename)

                        self.log('Image saved at %s' % full_filename)

                        # Release image
                        image_result.Release()
                        # self.log(f'Released image {i}')
                        i += 1
                except PySpin.SpinnakerException as ex:
                    self.log(f'Error acquiring single image: {ex}', logging.ERROR)
                    self.__busy = False
                    return False
                    # self.execute_trigger()
                # previous_seconds = current_seconds

            else:
                for i in range(self._number_of_images):
                    # current_seconds = time.time()
                    # if (current_seconds - previous_seconds) >= 1.0:
                    try:
                        # self.grab_next_image_by_trigger()
                        if i == 0:
                            timeout = 5000
                        else:
                            timeout = fast_timeout
                        image_result = self._cam.GetNextImage(timeout)
                        # image_result = self.safe_grab(timeout=timeout)
                        if image_result.IsIncomplete():
                            self.log('Image incomplete with image status %d...' % image_result.GetImageStatus(),
                                     logging.WARNING)

                        else:
                            chunk_data = image_result.GetChunkData()
                            frame_id = chunk_data.GetFrameID()
                            timestamp = chunk_data.GetTimestamp()
                            # Print image information
                            if self.debug:
                                width = image_result.GetWidth()
                                height = image_result.GetHeight()
                                self.log(
                                    'Grabbed Image %d/%d, width = %d, height = %d' % (
                                    i+1, self._number_of_images, width, height))

                            # Convert image to Mono8
                            image_converted = processor.Convert(image_result, PySpin.PixelFormat_Mono8)

                            # Create a unique filename
                            # filename = '%s-%d.jpg' % (image_prefix, i+1)
                            filename = '%s-%d-%s.jpg' % (image_prefix, i+1, timestamp)
                            full_filename = os.path.join(self._path_to_images, filename)

                            # Save image
                            image_converted.Save(full_filename)

                            self.log('(%d/%d) Image saved at %s' % (i+1, self._number_of_images, full_filename) )

                            # Release image
                            image_result.Release()
                            # self.log(f'Released image {i}')
                            i += 1
                    except PySpin.SpinnakerException as ex:
                        self.log(f'Error acquiring images: {ex}', logging.ERROR)
                        self.__busy = False
                        return False
                        # self.execute_trigger()
                    # previous_seconds = current_seconds
            self._cam.EndAcquisition()
            time.sleep(0.5)
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            self.__busy = False
            return False
        # self.acquisition_mode = PySpin.AcquisitionMode_Continuous
        # self.log('Acquisition mode set back to continuous...')
        self.__busy = False
        return True

    def safe_grab(self, timeout=1000, attempts=0) -> PySpin.Image:
        try:
            image_result = self._cam.GetNextImage(timeout)
        except PySpin.SpinnakerException as ex:
            logging.warning(f'Error: {ex}')
            if attempts < 3:
                attempts += 1
                return self.safe_grab(timeout=timeout, attempts=attempts)
            else:
                raise ex
        return image_result

    def execute_trigger(self):
        try:
            if self._chosen_trigger == TriggerType.SOFTWARE:
                if self._cam.TriggerSoftware.GetAccessMode() != PySpin.WO:
                    raise PySpin.SpinnakerException('Unable to execute trigger. Aborting...')
                self._cam.TriggerSoftware.Execute()
                # TODO: Blackfly and Flea3 GEV cameras need 2 second delay after software trigger
            elif self._chosen_trigger == TriggerType.HARDWARE:
                self.log('Use the hardware to trigger image acquisition.')
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return

    @property
    def device_serial_number(self) -> str:
        try:
            if self._cam.DeviceSerialNumber.GetAccessMode() < PySpin.RO:
                self.log('Unable to read the serial number', logging.WARNING)
                return ''
            serial_number = str(self._cam.DeviceSerialNumber.GetValue())
        except PySpin.SpinnakerException as ex:
            self.log(f'Error: {ex}', logging.ERROR)
            return ''
        return serial_number

    def set_logger(self, log: logging.Logger):
        if isinstance(log, logging.Logger):
            self._log = log

    def log(self, msg: str, level=logging.INFO):
        if self._log is not None:
            if isinstance(self._log, logging.Logger):
                self._log.log(level=level, msg=msg)
            else:
                print(msg)
        else:
            print(msg)

    def reset(self):
        self.reset_trigger()
        self.reset_frame_rate()
        self.reset_exposure()
        self.reset_gain()
        self.acquisition_mode = PySpin.AcquisitionMode_Continuous

    def shutdown(self):
        self.reset()
        self._cam.DeInit()
        del self._cam
        self._cam_list.Clear()
        self._system.ReleaseInstance()
        self.log(msg='Deleted camera instance.', level=logging.INFO)

    def __del__(self):
        self.shutdown()
