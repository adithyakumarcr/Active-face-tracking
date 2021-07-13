# Active Face Detection


#### Introduction

This program is developed to actively track human face and enable to self adjust to center. This is exclusivly developed for Jetson nano using webcam. The program can be modified for CSI camera. It is also programmed to manually take control of the servomotors using keyboard keys for movements.


A PID Controller is used to control the 2 servomotors that were used in the project. 


Default arguments used:
    1. Height and width are set to (480,640)
    2. and Pitch and Yaw are set to (60,90)


PID Constants used,
For Motor 1: Kd:0.75  Kp: 0.4 Ki: 0.1
For Motor 2: Kd: 0.05 Kp: 0.4 Ki: 0.08

(PID Constant needs optimisation based on which motor is being used)


### Arguments to launch the program

- **--resolution** To set the resolution (Width, Height) Default [640,480]
- **--deviceid** (For Windows only) To set the USB CameraID Default [0](Internal webcam)
- **--COMPORT** (For Windows) COM(int) and Ubuntu, please look into Ardiuno for serial portID Default("/dev/ttyACM0",9600)

### Hardware Required

1. 2x Servomotors
2. Arduino board (Micro used)
3. 1A Power adapter for powering servo's
4. Jumperwires
5. Jetson Nano with Jetpack (4.2+)


### Libraries Required

1. OpenCV  
2. Keyboard  
3. Timer  
4. Serial  

### Additional Features Include:

1. Manual overide control over servomotors  
2. FPS Counter
3. Face recognition using Facenets(Development Progess)
