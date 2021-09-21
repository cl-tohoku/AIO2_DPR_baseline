# path
WORK_DIR=${path to working directory}
# Train Reader =============================

TRAIN_EPOCH=50
DEV_EPOCH=1

TRAIN_BATCH=8
DEV_BATCH=4

SEQ_LEN=256
MAX_GRAD_NORM=2.0
WARMUP_STEP=1237
LEARNING_RATE=2e-5
GRADIENT_ACCUMULATION_STEP=1
DROPOUT=0.2
SEED=1
PASSAGE_PER_QUESTION=24
PASSAGE_PER_QUESTION_PREDICT=100

DATE=`date +%Y%m%d-%H%M`
# Train Reader ======================================
encoder_params="
--sequence_length $SEQ_LEN
--encoder_model_type hf_bert
--pretrained_model_cfg cl-tohoku/bert-base-japanese-whole-word-masking
"

training_params="
--num_train_epochs $TRAIN_EPOCH
--batch_size $TRAIN_BATCH
--dev_batch_size $DEV_BATCH
--max_grad_norm $MAX_GRAD_NORM
--warmup_steps $WARMUP_STEP
--learning_rate $LEARNING_RATE
--dropout $DROPOUT
--gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEP
--seed $SEED
--passages_per_question $PASSAGE_PER_QUESTION
--passages_per_question_predict $PASSAGE_PER_QUESTION_PREDICT
"

FI_TEST=${path tp test dataset}
OUT_DIR=${WORK_DIR}/outputs
MODEL_DIR=${OUT_DIR}/models
PREDICTION_RESULTS_DIR=${OUT_DIR}/prediction_results
TENSORBOARD_DIR=${OUT_DIR}/tensorboard_dir
mkdir -p ${OUT_DIR} ${MODEL_DIR} ${PREDICTION_RESULTS_DIR} ${TENSORBOARD_DIR}

CUDA_VISIBLE_DEVICES=0 \
python ${WORK_DIR}/src/reader_train.py \
    ${encoder_params} \
    ${training_params} \
    --fp16 \
    --dev_file ${FI_TEST} \
    --output_dir ${MODEL_DIR} \
    --dir_tensorboard ${TENSORBOARD_DIR} \
    --loss_and_score_results_dir ${OUT_DIR} \
| tee ${OUT_DIR}/reader_train_${DATE}.log