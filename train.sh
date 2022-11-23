python run_summarization.py \
    --model_name_or_path google/mt5-small \
    --do_train \
    --do_eval \
    --train_file data/train.jsonl \
    --validation_file data/public.jsonl \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --per_device_eval_batch_size 4 \
    --eval_accumulation_steps 4 \
    --predict_with_generate \
    --text_column maintext \
    --summary_column title \
    --adafactor \
    --learning_rate 1e-3 \
    --warmup_ratio 0.1 \
    --eval_steps 200 \
    --save_steps 200 \
    --logging_steps 200 \
    --evaluation_strategy steps \
    --num_beams 1 \
    --load_best_model_at_end True  \
    --save_total_limit 20 \
    --ignore_pad_token_for_loss False \
    --report_to all \
    --overwrite_output_dir \
    --output_dir mt5-small/greedy \
