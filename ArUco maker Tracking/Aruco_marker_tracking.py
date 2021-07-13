import numpy as np
import cv2
import cv2.aruco as aruco
import glob
import math
import struct
import serial

#Serial port communication with Arduino 
ser=serial.Serial('COM4',9600)


camera_device_id=1


markerLength = 0.25

cap = cv2.VideoCapture(camera_device_id)
width = 640
height = 480

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


Target_center_x=width/2
Target_center_y=height/2

#PID Controller constants
kd_x=0.75
Kp_x=0.4
ki_x=0.1

kd_y=0.05
Kp_y=0.4
ki_y=0.08

p_error_x=0
p_error_y=0

pitch=60
yaw=90


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*7,3), np.float32)
objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

objpoints = [] 
imgpoints = []

images = glob.glob('calib_images/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (7,6),None)

    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)


ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)

calibrationFile = "calibrationFileName.xml"
calibrationParams = cv2.FileStorage(calibrationFile, cv2.FILE_STORAGE_READ) 


camera_matrix = calibrationParams.getNode("cameraMatrix").mat() 
dist_coeffs = calibrationParams.getNode("distCoeffs").mat() 
count = 1



while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    arucoParameters =  aruco.DetectorParameters_create()
    aruco_list = {}
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=arucoParameters)
    
    if np.all(ids != None):
        if len(corners):
            for k in range(len(corners)):
                temp_1 = corners[k]
                temp_1 = temp_1[0]
                temp_2 = ids[k]
                temp_2 = temp_2[0]
                aruco_list[temp_2] = temp_1
        key_list = aruco_list.keys()
        font = cv2.FONT_HERSHEY_SIMPLEX
        for key in key_list:
            dict_entry = aruco_list[key]    
            centre = dict_entry[0] + dict_entry[1] + dict_entry[2] + dict_entry[3]
            centre[:] = [int(x / 4) for x in centre]
            orient_centre = centre + [0.0,5.0]
            centre = tuple(centre)  
            orient_centre = tuple((dict_entry[0]+dict_entry[1])/2)
            print("X:",centre[0])
            print("Y:",centre[1])
            x_center=centre[0]
            y_center=centre[1]

            cv2.circle(frame,centre,1,(0,0,255),8)

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
            p_error_x=error_x

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

            p_error_y=error_y
        
        else:
            pass

        ser.write(struct.pack('>BB',yaw,pitch))
            
        display = aruco.drawDetectedMarkers(frame, corners)
        display = np.array(display)
    else:
        display = frame

    cv2.imshow('Display',display)



    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()