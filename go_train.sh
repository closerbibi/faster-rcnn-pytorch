rm ./data/cache/*.pkl
./train.py 2>&1 | tee ./log/log_resnet101-bn-block1-fix-sgd.txt
#./train.py #2>&1 | tee ./log/log_trying.txt
