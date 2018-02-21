CURR_EXP="500k_tkn/uni_0.25"
echo "Building $CURR_EXP Model";

python "$BUILD_DIR/main_lm.py" train "$DATA_DIR/$CURR_EXP" "$MODEL_DIR/$CURR_EXP" \
--sentence_level --log_level debug  --training_weights True \
--load_model_opt "$MODEL_DIR/config/model_opt.json" \
--load_train_opt "$MODEL_DIR/config/train_opt.json" \
--batch_size 10;
echo "$CURR_EXP Model Training Complete";

#python ~/IWAL_experiments/seqmodel/main_lm.py eval "$DATA_DIR/$CURR_EXP" "$MODEL_DIR/$CURR_EXP" \
#--gpu --batch_size=1 --sentence_level --eval_latest \
#--load_model_opt "$MODEL_DIR/config/model_opt.json" \
#--load_train_opt "$MODEL_DIR/config/train_opt.json" ;
