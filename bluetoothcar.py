import asyncio
import threading
import time

import numpy as np
from bleak import BleakScanner
from bleak import BleakClient
from scipy.signal import resample
from brainflow.board_shim import BoardShim, BrainFlowInputParams, LogLevels, BoardIds
from brainflow.data_filter import DataFilter, DetrendOperations, FilterTypes
from inference import EEGInferenceApp
import matplotlib.pyplot as plt

RC_CAR_CHARACTERISTIC = '19b10000-e8f2-537e-4f6c-d104768a1214'

class BluetoothCar:
    LEFT = 1
    RIGHT = 2

    def __init__(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        def run_loop():
            asyncio.run(self._coroutine())
            print("Run loop exited")
        
        self.thread = threading.Thread(target=run_loop)
        self.command = 0
        self.client = None
        self.running = False

    def start(self):
        if (self.running == False):
            self.running = True
            self.thread.start()

    def stop(self):
        if (self.running == True):
            self.running = False
            self.thread.join()

    def move(self, direction):
        print(f"Move {direction}")
        self.command = direction

    # This is our asyncio loop running in a thread
    async def _coroutine(self):
        await self._connect_car()

        while (self.running):
            if (self.command > 0): 
                await self._move_car(self.command)
                self.command = 0

            await asyncio.sleep(0.3)

        print("Exited")
        
    async def _connect_car(self):
        print("Searching Arduino R4, please wait...")
        devices = await BleakScanner.discover(timeout=5, return_adv=True)
        for ble_device, adv_data in devices.values():
            if ble_device.name == 'RC Car':
                print("RC Car device found")
                self.client = BleakClient(ble_device.address)
                await self.client.connect()
                print("Connected to RC Car")

    async def _move_car(self, command):
        if (self.client == None):
            print("Car not connected!")
            return
    
        if command == BluetoothCar.LEFT:
            print("Move car left")
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x02", response=True)
            await asyncio.sleep(1)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x00", response=True)
            await asyncio.sleep(1)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x01", response=True)
            await asyncio.sleep(0.75)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x00", response=True)
        elif command == BluetoothCar.RIGHT:
            print("Move car right")
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x04", response=True)
            await asyncio.sleep(1)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x00", response=True)
            await asyncio.sleep(1)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x01", response=True)
            await asyncio.sleep(0.75)
            await self.client.write_gatt_char(RC_CAR_CHARACTERISTIC, b"\x00", response=True)

# Debug code
def main():
    car = BluetoothCar()
    car.start()
    time.sleep(5)
    
    car.move(BluetoothCar.LEFT)
    time.sleep(10)
    car.move(BluetoothCar.RIGHT)
    time.sleep(10)
    car.move(BluetoothCar.RIGHT)

    time.sleep(10)
    car.stop()
    
if __name__ == "__main__":
    main()
