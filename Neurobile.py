from pyqtgraph.Qt import QtWidgets, QtCore
import pyqtgraph as pg
import time
import pyttsx3
import winsound
import threading
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import resample
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations
from inference import EEGInferenceApp
from bluetoothcar import BluetoothCar

class DataProcessor:
    STATE_WAITING_FOR_BEEP = 0
    STATE_CAPTURING_DATA = 1

    def __init__(self, board_shim):
        self.board_id = board_shim.get_board_id()
        self.board_shim = board_shim
        self.exg_channels = BoardShim.get_exg_channels(self.board_id)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)
        self.update_speed_ms = 100
        self.window_size = 3
        self.num_points = self.window_size * self.sampling_rate
        self.speech = pyttsx3.init()

        # BCI variables
        self.last_beep_time_ms = 0
        self.beep_interval_ms = 10000
        self.data_capture_length_ms = 3000
        self.state = self.STATE_WAITING_FOR_BEEP
        self.model = EEGInferenceApp()
        self.car = BluetoothCar()
        self.car.start()

        # Initialize PyQt application
        self.app = QtWidgets.QApplication([])
        self.win = pg.GraphicsLayoutWidget(show=True, title='Cyton EEG Plot')
        self.win.resize(800, 600)

        # Initialize time series plots
        self._init_timeseries()

        # Start timer for updating plots
        timer = QtCore.QTimer()
        timer.timeout.connect(self.update)
        timer.start(self.update_speed_ms)
        QtWidgets.QApplication.instance().exec_()

    def __del__(self):
        if hasattr(self, 'car'):
            self.car.stop()
        if hasattr(self, 'app'):
            self.app.exit()

    def _init_timeseries(self):
        self.plots = []
        self.curves = []
        for i in range(len(self.exg_channels[:3])):
            p = self.win.addPlot(row=i, col=0)
            p.showAxis('left', False)
            p.setMenuEnabled('left', False)
            p.showAxis('bottom', False)
            p.setMenuEnabled('bottom', False)
            if i == 0:
                p.setTitle('C3')
            elif i == 1:
                p.setTitle('C4')
            elif i == 2:
                p.setTitle('CZ')
            self.plots.append(p)
            curve = p.plot()
            self.curves.append(curve)

    def update(self):
        trim = 50
        try:
            # Get current EEG data
            raw_data = self.board_shim.get_current_board_data(self.num_points + trim * 2)
            if (len(raw_data[1]) != self.num_points + trim * 2):
                return

            data = np.zeros((3, 384)) # 3 channels x 3s of 128 Hz EEG

            for count, channel in enumerate(self.exg_channels[:3]):
                channel_data = raw_data[channel][trim:-trim]

                # Apply detrending
                DataFilter.detrend(channel_data, DetrendOperations.CONSTANT.value)

                # Apply bandpass filter (4-50 Hz)
                DataFilter.perform_bandpass(
                    channel_data,
                    self.sampling_rate,
                    4.0,
                    50.0,
                    4,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE,
                    0,
                )

                # Remove mains noise 
                DataFilter.perform_bandstop(
                    channel_data,
                    self.sampling_rate,
                    58.0,
                    62.0,
                    4,
                    FilterTypes.BUTTERWORTH_ZERO_PHASE,
                    0,
                )

                # Resample to 128 Hz
                data[count] = resample(channel_data, int(len(channel_data) * 128 / self.sampling_rate), axis=0)

                # Scale to max +/- 50-100 uV
                max = np.max(data[count])
                data[count] *= (1/max/1e6 * 75)

                # Update the plot with filtered data
                self.curves[count].setData(data[count].tolist())
            
            self.processBCIEEG(data)
            self.app.processEvents()
        except Exception as e:
            print(f"Error during update: {e}")

    # This method gets called every update_speed_ms 
    def processBCIEEG(self, data):
        if (self.state == DataProcessor.STATE_WAITING_FOR_BEEP):
            if (self.current_time_ms() - self.last_beep_time_ms >= self.beep_interval_ms):
                self.last_beep_time_ms = self.current_time_ms()

                print("*** Think left or right ***")
                self.speak_async("Think left or right")
            
                # Change state to capturing 3s of EEG data
                self.state = DataProcessor.STATE_CAPTURING_DATA
                
        if (self.state == DataProcessor.STATE_CAPTURING_DATA):
            if (self.current_time_ms() - self.last_beep_time_ms > self.data_capture_length_ms):
                print(f"C3 max: {np.max(data[0])}, min: {np.min(data[0])}")
                
                #lines = plt.plot(data.T)
                #plt.legend(lines, ['EEG:C3', 'EEG:C4', 'EEG:Cz'])
                #plt.show()

                movement = self.model.predict_imagined_movement(data)
                if movement == EEGInferenceApp.LEFT:
                    print("Detected: left")
                    self.speak_async("You thought left")
                    time.sleep(0.3)
                    self.car.move(BluetoothCar.LEFT)
                else:
                    print("Detected: right")
                    self.speak_async("You thought right")
                    time.sleep(0.3)
                    self.car.move(BluetoothCar.RIGHT)

                # Change state back to waiting for the next beep at beep_interval_ms interval
                self.state = DataProcessor.STATE_WAITING_FOR_BEEP

    def speak_async(self, text):
        def _speak():
            self.speech.say(text)
            self.speech.runAndWait()
        threading.Thread(target=_speak).start()
    
    def play_beep_async(self):
        def _play_beep():
            winsound.Beep(frequency=1000, duration=700)
        threading.Thread(target=_play_beep).start()

    def current_time_ms(self):
        return round(time.time() * 1000)

def main():
    BoardShim.enable_dev_board_logger()

    # Set up BrainFlowInputParams for Cyton
    params = BrainFlowInputParams()
    params.serial_port = 'COM3'  # Replace 'COM3' with the correct port for your Cyton board

    # Initialize the Cyton board
    board_shim = BoardShim(BoardIds.CYTON_BOARD.value, params)
    try:
        board_shim.prepare_session()
        board_shim.start_stream(450000)
        DataProcessor(board_shim)
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        if board_shim.is_prepared():
            board_shim.release_session()

if __name__ == "__main__":
    main()
