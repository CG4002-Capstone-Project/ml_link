## Commands

Development
```
IP_ADDRESS=localhost EVAL_PORT=8000 DANCE_PORT=9000 IS_DASHBOARD=1 python3 ultra96.py
EVAL_PORT=8000 python3 eval_server.py
DANCE_PORT=9000 DANCER_ID=0 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=0 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=1 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=1 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=1 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=1 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=2 IS_DASHBOARD=1 IS_EMG=1 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=2 IS_DASHBOARD=1 IS_EMG=1 IS_POSITION=1 python3 laptop.py
```

Production
```
ssh -i "govtech.pem" ubuntu@ec2-52-221-205-129.ap-southeast-1.compute.amazonaws.com
ssh -L 9000:localhost:9000 -J e0315868@sunfire.comp.nus.edu.sg xilinx@makerslab-fpga-18
IP_ADDRESS=localhost EVAL_PORT=8000 DANCE_PORT=9000 IS_DASHBOARD=1 python3 ultra96.py
EVAL_PORT=8000 python3 eval_server.py
DANCE_PORT=9000 DANCER_ID=0 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=0 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=1 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=1 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=1 IS_DASHBOARD=1 IS_EMG=0 IS_POSITION=1 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=2 IS_DASHBOARD=1 IS_EMG=1 IS_POSITION=0 python3 laptop.py
DANCE_PORT=9000 DANCER_ID=2 IS_DASHBOARD=1 IS_EMG=1 IS_POSITION=1 python3 laptop.py
```