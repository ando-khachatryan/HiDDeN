nohup python -u unet_main.py new -d /data/coco/10K -e 300 -b 32 --noise "crop((0.2,0.25),(0.2,0.25))" --enc-loss-weight 5 --adv-loss-weight 0.005 --name unet-crop > unet-crop.log & 
sleep 1
tail -f unet-crop.log
