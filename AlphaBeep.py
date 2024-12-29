import time
import winsound

from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, WindowOperations, DetrendOperations
from brainflow.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes



def main():
    BoardShim.enable_dev_board_logger()

    # Setup the Cyton board
    params = BrainFlowInputParams()
    params.serial_port = "COM3"

    board_id = BoardIds.CYTON_BOARD.value
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)

    board = BoardShim(board_id, params)
    board.prepare_session()
    board.start_stream()
    BoardShim.log_message(LogLevels.LEVEL_INFO.value, 'start sleeping in the main thread')
    time.sleep(3)
    
    eeg_channels = board_descr['eeg_channels']
    eeg_channel = eeg_channels[0]
    
    while True:
        # Get board data and remove data from ringbuffer
        data = board.get_board_data()
        data_length = len(data[eeg_channel])
        print(f"Got data of length {data_length}")
        
        if (data_length < nfft ) : 
            print("Not enough data - ignoring")
            time.sleep(1.1)
            continue

        energy = 0
        for i in range(10):
            energy += abs(data[eeg_channel][i])
        energy /= 10
        print(f"Energy {energy}")

        DataFilter.perform_bandpass(data[eeg_channel], BoardShim.get_sampling_rate(board_id), 2.0, 40.0, 4,
                                        FilterTypes.BESSEL_ZERO_PHASE, 0)
 
        DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate,
                                        WindowOperations.BLACKMAN_HARRIS.value)

        band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
        band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
        power_ratio = band_power_alpha / band_power_beta

        print(f"alpha {band_power_alpha}")
        print(f"beta {band_power_beta}")
        print(f"alpha/beta {power_ratio}")

        if power_ratio > 5:
            winsound.Beep(frequency=440, duration=500)

        time.sleep(1.1)  # Need enough time to get samples, e.g. at 256 hz, it's 256 samples a second

    board.stop_stream()
    board.release_session()


if __name__ == "__main__":
    main()