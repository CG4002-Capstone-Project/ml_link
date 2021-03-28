## Commands

Debugging
```
python3 ultra96.py --ip_addr localhost --model_type dnn --main_dancer_id 0 --guest_dancer_id 2 --dancer_ids 0 
python3 extcomm.py --dancer_id 0
```

Production
```
ssh -L 9091:localhost:9091 -J e0315868@sunfire.comp.nus.edu.sg xilinx@makerslab-fpga-18
python3 ultra96.py --is_dashboard True --is_eval_server True --ip_addr ec2-52-221-205-129.ap-southeast-1.compute.amazonaws.com --model_type fpga --main_dancer_id 0 --guest_dancer_id 2 --dancer_ids 0 1 2
python3 extcomm.py --dancer_id 0
```