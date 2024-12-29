import keyboard
import asyncio
import time
import winsound

from bleak import BleakScanner, BleakClient
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, DetrendOperations, WindowOperations

RC_CAR_CHARACTERISTIC = '19b10000-e8f2-537e-4f6c-d104768a1214'

async def control_car(client, move_forward_signal):
    while True:
        if move_forward_signal[0]:  # Check if EEG signal triggers movement
            print("Moving forward")
            await client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x01", response=True)
        else:
            # Stop the car if the signal condition isn't met
            await client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x00", response=True)
        
        await asyncio.sleep(0.1)  # Adjust frequency of checks

async def main():
    # Set up the EEG
    params = BrainFlowInputParams()
    params.serial_port = "COM3"
    board_id = BoardIds.CYTON_BOARD.value
    board = BoardShim(board_id, params)
    board_descr = BoardShim.get_board_descr(board_id)
    sampling_rate = int(board_descr['sampling_rate'])
    nfft = DataFilter.get_nearest_power_of_two(sampling_rate)
    board.prepare_session()
    board.start_stream()
    eeg_channel = board_descr['eeg_channels'][0]
    
    move_forward_signal = [False]  # Shared flag to trigger car movement

    # Search for the Arduino device and establish BLE connection
    print("Searching Arduino R4, please wait...")
    devices = await BleakScanner.discover(timeout=5, return_adv=True)
    for ble_device, adv_data in devices.values():
        if ble_device.name == 'RC Car':
            print("RC Car device found")
            async with BleakClient(ble_device.address) as client:
                print("Connected to device")

                # Start both EEG monitoring and car control
                eeg_task = asyncio.create_task(monitor_eeg(board, eeg_channel, nfft, sampling_rate, move_forward_signal))
                await control_car(client, move_forward_signal)
                await eeg_task  # Wait until both tasks are complete

async def monitor_eeg(board, eeg_channel, nfft, sampling_rate, move_forward_signal):
    while True:
        data = board.get_board_data()
        if len(data[eeg_channel]) < nfft:
            await asyncio.sleep(1.1)
            continue

        # Process EEG data to get alpha and beta band powers
        DataFilter.perform_bandpass(data[eeg_channel], sampling_rate, 2.0, 40.0, 4, FilterTypes.BESSEL_ZERO_PHASE, 0)
        DataFilter.detrend(data[eeg_channel], DetrendOperations.LINEAR.value)
        psd = DataFilter.get_psd_welch(data[eeg_channel], nfft, nfft // 2, sampling_rate, WindowOperations.BLACKMAN_HARRIS.value)
        
        band_power_alpha = DataFilter.get_band_power(psd, 7.0, 13.0)
        band_power_beta = DataFilter.get_band_power(psd, 14.0, 30.0)
        power_ratio = band_power_alpha / band_power_beta

        print(f"alpha: {band_power_alpha}, beta: {band_power_beta}, ratio: {power_ratio}")

        # Update movement signal based on alpha/beta ratio threshold
        if 15 > power_ratio > 5:
            print("Alpha/Beta threshold met: Moving forward")
            move_forward_signal[0] = True
        else:
            move_forward_signal[0] = False

        await asyncio.sleep(1.1)  # Time to process next set of samples

if __name__ == "__main__":
    BoardShim.enable_dev_board_logger()
    asyncio.run(main())
