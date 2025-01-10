# Neurobile
A brain controlled bluetooth car

Notable Materials Used
- Open BCI Cyton biosensing board
- EEG electrode cap
- Arduino UNO R4

**The Car** 
--
To make the physical car, I sketched out a few designs, then used CAD and 3D printing to bring it to life. Through the process of 3D printing, I went through numerous iterations, each time improving on anything from aesthetics to functionality. The final design has an apparent resemblance to an F1 car. I also created some fake decals for the car to mimic that look you may see on an actual race track. For the technical side of it, I used an Arduino UNO R4 to control the car. I used the Python Bleak client to connect the Arduino to a laptop via bluetooth, and you can check the details about how I did that in the code. Below you can find a diagram for my circuit design.

<img src="https://github.com/joshuajamesburke/Neurobile/blob/main/car1.jpg" width=800/>
<img src="https://github.com/joshuajamesburke/Neurobile/blob/main/Circut.png" width=800/>

**First Iteration – Alpha Car** 
--
This iteration moves the car forward when its controller closes its eyes. I used Python paired with an EEG board to measure the occurrences of “Alpha Waves” (brainwaves from 8-12 Hz that appear when people are in a relaxed state/when people close their eyes). I divide this value by the occurrences of “Beta Waves” (Brainwaves from 12-30 Hz) as a grounding to output an “Alpha Score.” The Alpha Score is updated about every second. In the code, this Alpha Score is tweaked so it most accurately detects alpha waves. When the code detects alpha waves, it sends a signal to the car and moves it forward. If not, the car stays put.

Alpha Car Demo: https://youtube.com/shorts/9b1vt2CRSy8?feature=share

**Final Iteration – Neurobile** 
--
This iteration moves the car correspondingly to its user thinking left or right. It is a lot more complicated because it doesn’t simply detect different frequency brainwaves. The code first grabs the users EEG data over 3 seconds after being prompted to think either left or right. It inputs the code into a convolutional neural network trained on left and right EEG data (using a third library) and whichever signal the neural network outputs is sent to the car. The neural network detects a very slight de-synchronisation in the contralateral or opposite side of the motor cortex based on the direction you imagined. Since the signal is almost unnoticeable, the neural network ended up having an accuracy of almost 80% trained on perfect data for a handful of subjects, although when used on my untrained brainwave data, the accuracy was lower. As a next step, I want to train the model on the user's data, which should greatly improve the accuracy.

Final Demo: https://youtube.com/shorts/8y0yUy0MFGg?feature=share


<img src="https://github.com/joshuajamesburke/Neurobile/blob/main/josh.jpg" width=400/>


