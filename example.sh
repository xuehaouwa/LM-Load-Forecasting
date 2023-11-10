python run_hf_s2s.py \
    --model_name_or_path google_pegasus-xsum \
    --do_train \
    --seed=32 \
    --num_train_epochs=6 \
    --save_total_limit=1 \
    --train_file train.json \
    --validation_file val.json \
    --output_dir pegasus \
    --per_device_train_batch_size=6 \
    --overwrite_output_dir \
    --predict_with_generate

python run_inference.py \
      -t test.json \
      -m pegasus \
      -s pegasus_pred_horizon24 \
      -p 24 \
      --model_name pegasus