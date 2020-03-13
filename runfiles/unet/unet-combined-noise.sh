nohup python -u ./src/unet_main.py new -d ./data/medium -e 400 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()+quant()" --enc-loss-weight 4 --adv-loss-weight 0.001 --name unet-combined-noise-enc=4-adv=0.001 > unet-combined-noise-enc=4-adv=0.001.out &
sleep 1
tail -f unet-combined-noise-enc=4-adv=0.001.out
