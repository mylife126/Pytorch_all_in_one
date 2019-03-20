# python -u main.py \
#     --batch_size 256 \
#     --max_epoch 50 \
#     --lr 0.001 \
#     --train_root "hw2p2_check/train_data/medium" \
#     --val_root "hw2p2_check/validation_classification/medium" \
#     --test_root "hw2p2_check/test_classification/medium" \
#     --test_txt "test_order_classification.txt" \
#     --num_workers 8 \
#     --model "res50" \
#     --progress_freq 10
python -u main_classification.py \
    --batch_size 256 \
    --max_epoch 50 \
    --lr 0.0001 \
    --train_root "hw2p2_check/train_data/big" \
    --val_root "hw2p2_check/validation_classification/big" \
    --save_dir "saved_models/classification_big2" \
    --pred_dir "prediction/classification_big2" \
    --test_root "hw2p2_check/test_classification/medium" \
    --test_txt "test_order_classification.txt" \
    --checkpoint "saved_models/classification_big/para_epoch8.pkl" \
    --num_workers 8 \
    --model "res50" \
    --progress_freq 10
