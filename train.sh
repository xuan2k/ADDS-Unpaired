python train.py  --num_epoch=20 \
                 --split=oxford_unpaired \
                 --data_path=/content/drive/MyDrive/Dataset/oxfordrobocar \
                 --log_dir=checkpoints \
                 --model_name=adds_depth_unpaired \
                 --batch_size=8 \
                 --num_discriminator=3 \
                 --unpaired=True \
                 --smooth_domain_label