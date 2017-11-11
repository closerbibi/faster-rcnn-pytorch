rm ./data/cache/*.pkl
./train.py 2>&1 | tee ./log/log_resnet50-pooling.txt
#./train.py #2>&1 | tee ./log/log_trying.txt
