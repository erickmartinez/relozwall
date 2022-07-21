import logging
import os
import sys

sys.path.append('../')
sys.modules['cloudpickle'] = None

import time
from pymeasure.display.Qt import QtGui
from pymeasure.display.windows import ManagedWindow
from pymeasure.experiment import Procedure, Results
from pymeasure.experiment import FloatParameter, Parameter, IntegerParameter
from pymeasure.experiment import unique_filename
from instruments.esp32 import DualTCLoggerTCP
from instruments.inhibitor import WindowsInhibitor
from PyQt5.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QLabel

TC_LOGGER_IP = '192.168.4.3'


class TCTimeConstantProcedure(Procedure):
    thermocouple_id = Parameter("Thermocouple ID", default="TC1")
    measurement_time = FloatParameter("Measurement time", units='s', minimum=5.0, maximum=3600.0, default=10.0)

    __keep_alive: bool = False
    __on_sleep: WindowsInhibitor = None
    __tc_logger: DualTCLoggerTCP = None

    DATA_COLUMNS = ["Measurement Time (s)", "Temperature (C)"]

    def startup(self):
        log.info('Connecting to readout...')
        self.__tc_logger = DualTCLoggerTCP(ip_address=TC_LOGGER_IP)
        log.info('Connection successful!')

    def execute(self):
        log.info('Preparing the system...')
        self.__tc_logger.log_time = self.measurement_time


    def shutdown(self):
        if self.__tc_logger is not None:
            self.__tc_logger.close()
        # Remove file handlers from logger
        if len(log.handlers) > 0:
            for handler in log.handlers:
                if isinstance(handler, logging.FileHandler):
                    log.removeHandler(handler)


class CustomDialog(QDialog):
    def __init__(self, message_txt, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Wait a second.")

        QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(message_txt)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)


class MainWindow(ManagedWindow):

    def __init__(self):
        super(MainWindow, self).__init__(
            procedure_class=TCTimeConstantProcedure,
            inputs=['thermocouple_id', "measurement_time"],
            displays=['thermocouple_id', "measurement_time"],
            x_axis="Measurement Time (s)",
            y_axis="Temperature (C)",
            directory_input=True,
        )
        self.setWindowTitle('Thermocouple time constant measurement')

    def queue(self):
        directory = self.directory

        procedure: TCTimeConstantProcedure = self.make_procedure()
        thermocouple_id = procedure.thermocouple_id

        prefix = f'CAL_{thermocouple_id}_'
        filename = unique_filename(directory, prefix=prefix)
        log_file = os.path.splitext(filename)[0] + '.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        fh = logging.FileHandler(log_file)
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

        results = Results(procedure, filename)
        experiment = self.new_experiment(results)
        procedure.unique_filename = filename
        self.manager.queue(experiment)


if __name__ == "__main__":
    log = logging.getLogger(__name__)
    log.addHandler(logging.NullHandler())

    # create console handler and set level to debug
    has_console_handler = False
    if len(log.handlers) > 0:
        for handler in log.handlers:
            if isinstance(handler, logging.StreamHandler):
                has_console_handler = True

    if not has_console_handler:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        log.addHandler(ch)

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
