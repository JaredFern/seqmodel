echo "Extending ngram 500k Token Model";

python main_lm.py train "$DATA_DIR/ngram" "$MODEL_DIR/ngram" \
--gpu --sentence_level --log_level debug  --training_weights True \
--load_model_opt "$MODEL_DIR/config/model_opt.json" \
--load_train_opt "$MODEL_DIR/config/train_opt.json" ;

echo "500k Base Model Training Complete";
# python ~/seqmodel/main_lm.py eval "$DATA_DIR/$MODE" "$MODEL_DIR/$MODE" \
# --gpu --batch_size=1 --sentence_level;
