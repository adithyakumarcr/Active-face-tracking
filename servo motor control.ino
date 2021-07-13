/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>

Servo myservo,myservo1;  // create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position
int read_pos[1];


void setup() {
  myservo.attach(9);  // attaches the servo on pin 9 to the servo object
  myservo1.attach(6);
}

void loop() {
  
 while(Serial.available()>= 2){
    read_pos[0] = Serial.read();
    read_pos[1] = Serial.read();
    
    myservo.write(read_pos[0]);
    myservo1.write(read_pos[1]);
    }

}
