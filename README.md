## Commands

Debugging
```
python3 ultra96.py --ip_addr localhost --model_type dnn --dancer_ids 0 
```

Production
```
python3 ultra96.py --dashboard True --is_eval_server True --ip_addr localhost --model_type fpga --dancer_ids 0 1 2
```