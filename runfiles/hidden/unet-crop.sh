nohup python -u unet_main.py new -d ./data/coco/10K -e 200 -b 32 --noise 'crop((0.4,0.55),(0.4,0.55))' --name unet-crop > unet-crop.log & 
sleep 1
tail -f unet-crop.log
