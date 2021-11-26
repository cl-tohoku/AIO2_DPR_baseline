# Path to the trained models.
BIENCODER_FILE='model/biencoder.pt'
READER_FILE='model/reader.pt'
EMBEDDING_FILE='model/embedding.pickle'
PASSAGES_FILE='model/passages.tsv.gz'

RETRIEVER_OUTPUT_FILE='model/retriever_output.json'
READER_OUTPUT_DIR='model/reader_output'

# Get predictions for all questions in the input.
INPUT_FILE=$1
OUTPUT_FILE=$2

# Now run predictions on input file.
echo 'Retrieving passages.'
python dense_retriever.py \
    --model_file $BIENCODER_FILE \
    --ctx_file $PASSAGES_FILE \
    --encoded_ctx_file $EMBEDDING_FILE \
    --qa_file $INPUT_FILE \
    --out_file $RETRIEVER_OUTPUT_FILE \
    --n-docs 100 \
    --validation_workers 32 \
    --batch_size 64 \
    --projection_dim 768
echo 'Reading the retrieved passages.'
mkdir $READER_OUTPUT_DIR
python train_reader.py \
    --dev_file $RETRIEVER_OUTPUT_FILE \
    --model_file $READER_FILE \
    --prediction_results_dir $READER_OUTPUT_DIR
echo 'Formatting the prediction.'
cat $READER_OUTPUT_DIR/test_prediction_results.json|jq -c '.[]|{qid: .qid, question: .question, prediction: .predictions[0].prediction.text}' > $OUTPUT_FILE
