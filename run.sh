python run_summarization.py \
    --model_name_or_path mt5-small \
    --do_predict \
    --test_file ${1} \
    --predict_with_generate \
    --text_column maintext \
    --per_device_eval_batch_size 4 \
    --num_beams 4 \
    --output_file ${2} \
    --output_dir ./output/beams4 \
