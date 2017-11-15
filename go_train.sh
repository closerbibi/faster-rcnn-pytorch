rm ./data/cache/*.pkl
./train.py \
2>&1 | tee ./log/log_resnet101-pascalvoc.txt
