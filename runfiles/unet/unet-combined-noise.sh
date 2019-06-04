nohup python -u unet_main.py new -d /data/coco/10K -e 400 -b 32 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()+quant()" --name unet-combined-noise > unet-combined-noise.log & 
sleep 1
tail -f unet-combined-noise.log
