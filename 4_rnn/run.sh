
python -u main.py \
    --batch_size 8 \
    --max_epoch 3 \
    --train_data_path "hw3p2-data-V2/wsj0_train.npy" \
    --train_label_path "hw3p2-data-V2/wsj0_train_merged_labels.npy" \
    --dev_data_path "hw3p2-data-V2/wsj0_dev.npy" \
    --dev_label_path "hw3p2-data-V2/wsj0_dev_merged_labels.npy" \
    --test_data_path "hw3p2-data-V2/transformed_test_data.npy" \
    --num_workers 8