mkdir -p logs
mkdir -p data/models
mkdir -p data/vocab
for seed in $(seq 6 10); do
    for task in binary-flip rev copy en-de; do
        unbuffer python -u train.py --debug --task $task --epochs 30 --loss-coef 0.0 --seed "$seed" --attention uniform --tensorboard_log
        unbuffer python -u train.py --debug --task $task --epochs 30 --loss-coef 0.0 --seed "$seed" --attention no-attention --tensorboard_log
        echo completed the config $task, seed: "$seed";
    done;
done;
