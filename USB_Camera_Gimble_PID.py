import sys
import argparse
import subprocess
import os
import serial
import struct
import time
import multiprocessing
import keyboard
import cv2
import numpy as np
import time

import concurrent.futures

parser=argparse.ArgumentParser()
 
parser.add_argument("--resolution", nargs="+", default=[640,480])
parser.add_argument( "--deviceid", nargs="+", default=[0])
parser.add_argument( "--COMPORT", nargs="+", default=["/dev/ttyACM0",9600])

value=parser.parse_args()

video_dev=value.deviceid[0]

image_width=value.resolution[0]
image_height=value.resolution[1]


#Serial port connection with Arduino Micro 

ser=serial.Serial(value.COMPORT[0],value.COMPORT[1]) 


WINDOW_NAME = 'Active Face Tracking'

#Target center for face_detection 
Target_center_x=image_width/2
Target_center_y=image_height/2


#PID Controller constants
kd_x=0.75
Kp_x=0.4
ki_x=0.1

kd_y=0.05
Kp_y=0.4
ki_y=0.08


def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '
               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_window(width, height):
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, width, height)
    cv2.moveWindow(WINDOW_NAME, 0, 0)
    cv2.setWindowTitle(WINDOW_NAME, 'Camera Demo for Jetson')


def joystick(yaw,pitch):
    if keyboard.is_pressed('a') and pitch>0 and yaw<=180:
        yaw=yaw-1
        
    if keyboard.is_pressed('d') and pitch>=0 and yaw<180:
        yaw=yaw+1

    if keyboard.is_pressed('w') and pitch>0 and yaw<=180:
        pitch=pitch-1

    if keyboard.is_pressed('s') and pitch>=0 and yaw<180:
        pitch=pitch+1
    
    return yaw,pitch
    


def face_detect_init():
    face_cascade = cv2.CascadeClassifier(
        'E:\\Work\\Beagle Systems\\Jetson_nano\\Face Tracking\\haarcascade_frontalface_default.xml'
    )
    eye_cascade = cv2.CascadeClassifier(
        'E:\\Work\\Beagle Systems\\Jetson_nano\\Face Tracking\\haarcascade_eye.xml'
    )

    return face_cascade, eye_cascade


def face_detect(cap,img):
    [face_cascade, eye_cascade] =face_detect_init()
    gray =cv2. cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.3,5)  #location of the face when detected
    

    for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y : y + h, x : x + w]
                roi_color = img[y : y + h, x : x + w]
                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
    return img, face_cascade, gray
    
   
def controller(face_cascade,gray,yaw,pitch, p_error_x, p_error_y):
    faces=face_cascade.detectMultiScale(gray,1.3,5)  #location of the face when detected

    for (x,y,w,h) in faces:
        pass
        
    if len(faces)!=0:  
        x_center=x+(w/2)
        y_center=y+(h/2) 

        error_x= int(( x_center-Target_center_x)/18)     #No exact conversion for this
        error_y= int(( y_center-Target_center_y)/13)     #No relation for this

        if x_center!=Target_center_x:
            PID_p=Kp_x*error_x
            PID_d = kd_x*((error_x-p_error_x))

           #Integral Controller
            if x_center-Target_center_x<50 and x_center-Target_center_x>-50:
                PID_i=(ki_x*error_x)
            else:
                PID_i=0
                    
            yaw=yaw-int(PID_d+PID_p+PID_i)
            yaw= np.clip(yaw,10,180)
            #print("Error for X:",x_center-Target_center_x)
        else:
            pass
            
        if y_center!=Target_center_y:
            PID_py=Kp_y*error_y
            PID_dy=kd_y*((error_y-p_error_y))

            #Integral Controllerṇ
            if x_center-Target_center_x<50 and x_center-Target_center_x>-50:
                PID_i=(ki_y*error_x)
            else:
                PID_i=0

            pitch=pitch+int(PID_py+PID_dy+PID_i)
            pitch=np.clip(pitch,45,120)
            #print("Error for Y:", y_center-Target_center_y )
        else:
            pass
        
        ser.write(struct.pack('>BB',yaw,pitch)) 

    else:
        #sweep()
        error_x=0
        error_y=0
    
    return yaw,pitch,error_x,error_y

sweep_yaw =10
sweep_pitch =45

def sweep():

    global sweep_pitch
    global sweep_yaw
    sweep_yaw=sweep_yaw+1


    if sweep_yaw>=180:
        sweep_yaw=10
        sweep_pitch=sweep_pitch+30
    sweep_yaw=np.clip(sweep_yaw,10,180)
    sweep_pitch= np.clip(sweep_pitch, 45,120)
    ser.write(struct.pack('>BB',sweep_yaw,sweep_pitch))


def read_cam(cap,yaw,pitch):
    p_error_x=0
    p_error_y=0
    period=0.16
    
    while True:
        start_time=time.thread_time_ns()
        if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
            break
        _,img = cap.read() # grab the next image frame from camera
        image,face_cascade,gray=face_detect(cap,img) 
        yaw,pitch,error_x,error_y=controller(face_cascade,gray,yaw,pitch,p_error_x,p_error_y)
        
        p_error_x=error_x 
        p_error_y=error_y    

        cv2.imshow(WINDOW_NAME, image)
        key = cv2.waitKey(5)
        if key == 27: # ESC key: quit program
            break

        print("Time:", (time.thread_time_ns()-start_time)/1000000)
        print("\033c")





def write_cam():
    i=0
    while True:
        _,im2=cap.read()

        cv2.imshow(WINDOW_NAME, im2)
        key = cv2.waitKey(10)
        if key == 27: # ESC key: quit program
            path= '/home/dlinano/Images'
            cv2.imwrite(os.path.join(path, 'pic%3d.jpg'%(i)), im2)
            i=i+1
        elif key == ord('c') or key == ord ('C'):
            break



if __name__ == '__main__':
    pitch=60
    yaw=90
    
    ser.write(struct.pack('>BB',yaw,pitch))

    #For Jetson Nano
    elif(value.COMPORT[0]=="/dev/ttyACM0"):
        cap = open_cam_usb(video_dev, image_width, image_height)     

    #For Personal PC:
    if(value.COMPORT[0]=="COM11"):
        cap=cv2.VideoCapture(1)
    else:
        print("Enter COMPORT correctly")
    
    if not cap.isOpened():
        sys.exit('Failed to open camera!')
    
    open_window(image_width, image_height) 
    read_cam(cap,yaw,pitch)

    cap.release()
    cv2.destroyAllWindows()
