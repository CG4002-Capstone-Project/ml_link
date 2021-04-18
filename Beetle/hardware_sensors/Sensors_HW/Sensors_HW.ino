// I2C device class (I2Cdev) demonstration Arduino sketch for MPU6050 class using DMP (MotionApps v2.0)
// 6/21/2012 by Jeff Rowberg <jeff@rowberg.net>
// Updates should (hopefully) always be available at https://github.com/jrowberg/i2cdevlib
//
// Changelog:
//      2019-07-08 - Added Auto Calibration and offset generator
//       - and altered FIFO retrieval sequence to avoid using blocking code
//      2016-04-18 - Eliminated a potential infinite loop
//      2013-05-08 - added seamless Fastwire support
//                 - added note about gyro calibration
//      2012-06-21 - added note about Arduino 1.0.1 + Leonardo compatibility error
//      2012-06-20 - improved FIFO overflow handling and simplified read process
//      2012-06-19 - completely rearranged DMP initialization code and simplification
//      2012-06-13 - pull gyro and accel data from FIFO packet instead of reading directly
//      2012-06-09 - fix broken FIFO read sequence and change interrupt detection to RISING
//      2012-06-05 - add gravity-compensated initial reference frame acceleration output
//                 - add 3D math helper file to DMP6 example sketch
//                 - add Euler output and Yaw/Pitch/Roll output formats
//      2012-06-04 - remove accel offset clearing for better results (thanks Sungon Lee)
//      2012-06-01 - fixed gyro sensitivity to be 2000 deg/sec instead of 250
//      2012-05-30 - basic DMP initialization working

/* ============================================
I2Cdev device library code is placed under the MIT license
Copyright (c) 2012 Jeff Rowberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
===============================================
*/

// I2Cdev and MPU6050 must be installed as libraries, or else the .cpp/.h files
// for both classes must be in the include path of your project
#include "I2Cdev.h"

#include "MPU6050_6Axis_MotionApps20.h"
//#include "MPU6050.h" // not necessary if using MotionApps include file

// Arduino Wire library is required if I2Cdev I2CDEV_ARDUINO_WIRE implementation
// is used in I2Cdev.h
#if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
    #include "Wire.h"
#endif

#include <Thread.h>
#include <ThreadController.h>
#include <arduinoFFT.h>

MPU6050 mpu;

/* =========================================================================
   NOTE: In addition to connection 3.3v, GND, SDA, and SCL, this sketch
   depends on the MPU-6050's INT pin being connected to the Arduino's
   external interrupt #0 pin. On the Arduino Uno and Mega 2560, this is
   digital I/O pin 2.
 * ========================================================================= */

/* =========================================================================
   NOTE: Arduino v1.0.1 with the Leonardo board generates a compile error
   when using Serial.write(buf, len). The Teapot output uses this method.
   The solution requires a modification to the Arduino USBAPI.h file, which
   is fortunately simple, but annoying. This will be fixed in the next IDE
   release. For more info, see these links:

   http://arduino.cc/forum/index.php/topic,109987.0.html
   http://code.google.com/p/arduino/issues/detail?id=958
 * ========================================================================= */

// MPU control/status vars
bool dmpReady = false;  // set true if DMP init was successful
uint8_t devStatus;      // return status after each device operation (0 = success, !0 = error)
uint16_t packetSize;    // expected DMP packet size (default is 42 bytes)
uint16_t fifoCount;     // count of all bytes currently in FIFO
uint8_t fifoBuffer[64]; // FIFO storage buffer

// orientation/motion vars
Quaternion q;           // [w, x, y, z]         quaternion container
VectorInt16 aa;         // [x, y, z]            accel sensor measurements
VectorInt16 aaReal;     // [x, y, z]            gravity-free accel sensor measurements
VectorFloat gravity;    // [x, y, z]            gravity vector
float ypr[3];           // [yaw, pitch, roll]   yaw/pitch/roll container and gravity vector

//new tw
VectorInt16 gyro;         // [x, y, z]            gyro sensor measurements

//new tw
enum BeetleStates {IDLING, DANCING};    //three states
BeetleStates bState = IDLING; //Start state is Idling

int count_idling = 0;
int count_dancing = 0;
float total_acceleration = 0;
float abs_yaw = 0;
float abs_pitch = 0;
float abs_roll = 0;

//new tw, Threading
ThreadController t_controller = ThreadController();
Thread mainThread = Thread();
Thread emgThread = Thread();

arduinoFFT FFT = arduinoFFT(); /* Create FFT object */


//new tw
const int analogInPin = A0;  // Analog input pin that the potentiometer is attached to
//const int analogOutPin = 9; // Analog output pin that the LED is attached to

int sensorValue = 0;        // value read from the pot

/*
These values can be changed in order to evaluate the functions
*/
const uint16_t samples = 64; //This value MUST ALWAYS be a power of 2
const double samplingFrequency = 1000; //5000
/*
These are the input and output vectors
Input vectors receive computed results from FFT
*/
double vReal[samples];//= {0.862209, 0.43418, 0.216947544, 0.14497645}; //= {0, 0.862209, 0.43418, 0.216947544, 0.14497645, 0, 0, 0, 0, 0, 0, 0 ,0 ,0, 0, 0};//{0.63,0.06,0.26,0.06,0.11,0.05,0.04,0.06,0.02,0.05,0.01,0.05,0.01,0.05,0.01,0.05};
double vImag[samples]; 

bool sample_ready = false;
int s_count = 0; //sample count

long overall_s_count = 0; //total count from start to end
double accu_mav = 0; //Sum of all x values from start to end
double accu_rms = 0; //To be used for RMS

double mav = 0;
double rms = 0;

double mean_freq = 0;
double sum_freq_power = 0; //summation of freq*Power 
double sum_power = 0; //summation of Power

//new tw
// ================================================================
// ===                      STATE MACHINE                       ===
// ================================================================
int readValues () {
  if (total_acceleration > 0.53 && ((abs_pitch > 10) || (abs_roll > 10))) {
     return 1; //DANCING at this moment
  }
  return 0; //IDLING at this moment
}

bool isDancing_2() {                    //current state is idling, check if it is dancing
  if (readValues() == 1) {           //read DANCING
    count_dancing++;
    if (count_dancing >= 3) {
      return true;                   //Finally, idle->dancing
    }
  } else {                           //read IDLING, dancing is false alarm
    count_dancing = 0;
  }
                                        
  return false;                     //return false
}

bool isIdling() {                    //current state is moving/dancing, check if it is idling
  if (readValues() == 0) {           //read IDLING
    count_idling++;
    if (count_idling >= 4) {
      return true;                   //Finally, move->idle or dance->idle
    }
  } else {                          //read MOVING, idling is false alarm
    count_idling = 0;
  }                                      
  return false;                     //return false
}

void checkBeetleState() {         
  switch (bState) {              //Depending on the state
    case IDLING: {                   
      if (isDancing_2()) {            //Check if it is dancing
        bState = DANCING;
        count_idling = 0;
        //count_moving = 0;
        count_dancing = 0;
      }          
      break;                         
    }
    case DANCING: {              
      if (isIdling()) {            //Check if it is idling
         bState = IDLING;             
         count_idling = 0;
         //count_moving = 0;
         count_dancing = 0;
      }
       break;                         
    }
  }
}


// ================================================================
// ===                      THREAD FUNCTIONS                    ===
// ================================================================
void mainCallback(){

    mpu.resetFIFO();
    fifoCount = mpu.getFIFOCount();
    while (fifoCount < packetSize) fifoCount = mpu.getFIFOCount();

      // read a packet from FIFO
    while(fifoCount >= packetSize){ // Lets catch up to NOW, someone is using the dreaded delay()!
      mpu.getFIFOBytes(fifoBuffer, packetSize);
      // track FIFO count here in case there is > 1 packet available
      // (this lets us immediately read more without waiting for an interrupt)
      fifoCount -= packetSize;
    }

    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    //Serial.println("Edited!!");
    Serial.print("|ypr|");
    Serial.print(ypr[0] * 180/M_PI);
    Serial.print("|");
    Serial.print(ypr[1] * 180/M_PI);
    Serial.print("|");
    Serial.print(ypr[2] * 180/M_PI);
    Serial.println("|");
    
    mpu.dmpGetAccel(&aa, fifoBuffer);
    mpu.dmpGetLinearAccel(&aaReal, &aa, &gravity);
    Serial.print("|areal|");
    Serial.print(aaReal.x);
    Serial.print("|");
    Serial.print(aaReal.y);
    Serial.print("|");
    Serial.print(aaReal.z);
    Serial.println("|");

    mpu.dmpGetGyro(&gyro, fifoBuffer);
    Serial.print("|gyro|");
    Serial.print(gyro.x);
    Serial.print("|");
    Serial.print(gyro.y);
    Serial.print("|");
    Serial.print(gyro.z);
    Serial.println("|");
       
    total_acceleration = sqrt(pow(aaReal.x/ 8192.0,2)+ pow(aaReal.y/ 8192.0,2)+ pow(aaReal.z/ 8192.0,2));
    Serial.print("\t\t\t|total accel|");
    Serial.print(total_acceleration);
    Serial.println("|");

    abs_yaw = abs(ypr[0] * 180/M_PI);
    abs_pitch = abs(ypr[1] * 180/M_PI);
    abs_roll = abs(ypr[2] * 180/M_PI);
    Serial.print("|Abs ypr|");
    Serial.print(abs_yaw);
    Serial.print("|");
    Serial.print(abs_pitch);
    Serial.print("|");
    Serial.print(abs_roll);
    Serial.println("|");

    //new tw
    checkBeetleState();
    
    if (bState == DANCING) {
      Serial.println("\t\t\t\t|DANCING|");
    } else {         
      Serial.println("\t\t\t\t|IDLING|");
    }
    
    //Send emg values from here
    Serial.print("\t\t\t\tMAV = ");
    Serial.println(mav);
    Serial.print("\t\t\t\tRMS = ");
    Serial.println(rms);
    Serial.print("\t\t\t\tMean Freq = ");
    Serial.println(mean_freq);
    Serial.println("//////////////////////////"); 
}

void emgCallback(){
  if (!sample_ready) {
      sensorValue = analogRead(analogInPin);
      float voltage = sensorValue * (5.0 / 1023.0);
      vReal[s_count] = voltage; //store x values
      
      s_count = s_count+1;
    
      if (s_count == samples){
        sample_ready = true;
        s_count = 0;
      }
  } else { 
      
      for (int i = 0; i < samples; i ++) {       
        accu_mav += abs(vReal[i]);
        accu_rms += pow(vReal[i],2);
        vImag[i] = 0; //Important, prevent ovf
      }

      overall_s_count += samples; //include new samples
      mav = accu_mav*1.0/overall_s_count;
      rms = sqrt(accu_rms*1.0/overall_s_count);

      //Only Fourier transform AFTER getting MAV and RMS, due to reusing array
      FFT.Compute(vReal, vImag, samples, FFT_FORWARD); /* Compute FFT */
      
      for (int i = 0; i < (samples/2)+1; i++)
      {
        double vPower = pow(vReal[i],2) + pow(vImag[i],2) / (samples*samplingFrequency);

        if (i >= 1 && i < (samples/2))
        {
          vPower = 2*vPower;
        }
        
        sum_freq_power += (i*1.0*samplingFrequency/samples)*vPower;
        sum_power += vPower;
      }

      mean_freq = sum_freq_power*1.0/sum_power;

      sample_ready = false;
  }
}
// ================================================================
// ===                      INITIAL SETUP                       ===
// ================================================================

void setup() {
    // join I2C bus (I2Cdev library doesn't do this automatically)
    #if I2CDEV_IMPLEMENTATION == I2CDEV_ARDUINO_WIRE
        Wire.begin();
        Wire.setClock(400000); // 400kHz I2C clock. Comment this line if having compilation difficulties
    #elif I2CDEV_IMPLEMENTATION == I2CDEV_BUILTIN_FASTWIRE
        Fastwire::setup(400, true);
    #endif

    // initialize serial communication
    // (115200 chosen because it is required for Teapot Demo output, but it's
    // really up to you depending on your project)
    Serial.begin(115200);
    while (!Serial); // wait for Leonardo enumeration, others continue immediately

    // NOTE: 8MHz or slower host processors, like the Teensy @ 3.3V or Arduino
    // Pro Mini running at 3.3V, cannot handle this baud rate reliably due to
    // the baud timing being too misaligned with processor ticks. You must use
    // 38400 or slower in these cases, or use some kind of external separate
    // crystal solution for the UART timer.

    // initialize device

    Serial.println(F("Initializing I2C devices..."));
    mpu.initialize();
    //pinMode(INTERRUPT_PIN, INPUT);

    // verify connection
    Serial.println(F("Testing device connections..."));
    Serial.println(mpu.testConnection() ? F("!MPU6050 connection successful") : F("!MPU6050 connection failed"));

    // wait for ready
    Serial.println(F("\n!Send any character to begin DMP programming and demo: "));
    while (Serial.available() && Serial.read()); // empty buffer
    while (!Serial.available());                 // wait for data
    while (Serial.available() && Serial.read()); // empty buffer again

    // load and configure the DMP
    Serial.println(F("!Initializing DMP..."));
    devStatus = mpu.dmpInitialize();    

    // make sure it worked (returns 0 if so)
    if (devStatus == 0) {
        // Calibration Time: generate offsets and calibrate our MPU6050
        mpu.CalibrateAccel(6); //new tw 
        mpu.CalibrateGyro(6); //new tw
        mpu.PrintActiveOffsets();
        // turn on the DMP, now that it's ready
        Serial.println(F("Enabling DMP..."));
        mpu.setDMPEnabled(true);

        // enable Arduino interrupt detection
        //Serial.print(F("Enabling interrupt detection (Arduino external interrupt "));
        //Serial.print(digitalPinToInterrupt(INTERRUPT_PIN));
        Serial.println(F(")..."));
        //attachInterrupt(digitalPinToInterrupt(INTERRUPT_PIN), dmpDataReady, RISING);
        //mpuIntStatus = mpu.getIntStatus();

        // set our DMP Ready flag so the main loop() function knows it's okay to use it
        Serial.println(F("DMP ready! Waiting for first interrupt..."));
        dmpReady = true;

        // get expected DMP packet size for later comparison
        packetSize = mpu.dmpGetFIFOPacketSize();
    } else {
        // ERROR!
        // 1 = initial memory load failed
        // 2 = DMP configuration updates failed
        // (if it's going to break, usually the code will be 1)
        Serial.print(F("!DMP Initialization failed (code "));
        Serial.print(devStatus);
        Serial.println(F(")"));
    }

    // Configure 
    mainThread.onRun(mainCallback);
    mainThread.setInterval(27); 
 
    emgThread.onRun(emgCallback);
    emgThread.setInterval(1); //delay(1);
  
    // Adds both threads to the controller
    t_controller.add(&mainThread);
    t_controller.add(&emgThread); // & to pass the pointer to it
}



// ================================================================
// ===                    MAIN PROGRAM LOOP                     ===
// ================================================================
void loop() {
    
    // if programming failed, don't try to do anything
    if (!dmpReady) return;

    t_controller.run();
}
