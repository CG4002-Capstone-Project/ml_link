//Edited by Ting Wei
Source code is in Sensors_HW folder

Library files are arduinothread, arduinoFFt, and MPU6050, drag and drop them in
the C:\..........\Documents\Arduino\libraries


Changes:
2 states instead of 3 (idling and dancing)

threshold revamped to changing states from dancing to idling in approx. 1.1 to 1.7seconds
								instead of >2.5 seconds
Emg values are edited to work better than last time

