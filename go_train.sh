rm ./data/cache/*.pkl
./train.py 2>&1 | tee ./log/log_more_anchor_size.txt
#./train.py #2>&1 | tee ./log/log_trying.txt
