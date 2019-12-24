export CUDA_VISIBLE_DEVICES=0
for((i=0;i<2;i++));
do

python train.py \
--counter $i \
--name bertt128_bertwwm512 \
--model 0 \
--model1 1 \
--title_len 128 \
--content_len 512 \
--learning_rate 5e-5 \
--min_learning_rate 1e-5 \
--random_seed 123 \
--batch_size 16 \
--epoch 8 \
--fold 7

done

python combine.py --k 2




