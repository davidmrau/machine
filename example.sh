#! /bin/sh

TRAIN_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/train.tsv
DEV_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/validation.tsv
TEST_PATH=../machine-tasks/LookupTables/lookup-3bit/samples/sample1/new_compositions.tsv
EXPT_DIR=example

# set values
EMB_SIZE=128
H_SIZE=128
N_LAYERS=1
CELL='gru'
EPOCH=20
PRINT_EVERY=50
TF=0
BATCH_SIZE=1

# Start training
echo "Train model on example data"
python train_model.py --train $TRAIN_PATH --output_dir $EXPT_DIR --print_every $PRINT_EVERY --embedding_size $EMB_SIZE --hidden_size $H_SIZE --rnn_cell $CELL --n_layers $N_LAYERS --epoch $EPOCH --print_every $PRINT_EVERY --teacher_forcing $TF --attention 'pre-rnn' --attention_method 'mlp' --use_attention_loss --batch_size 1 --full_focus --save_every 10
echo "\n\nEvaluate model on test data"
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $DEV_PATH 
python evaluate.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1) --test_data $TEST_PATH
# echo "\n\nRun in inference mode"
# python infer.py --checkpoint_path $EXPT_DIR/$(ls -t $EXPT_DIR/ | head -1)
