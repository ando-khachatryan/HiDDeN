nohup python -u ./src/unet_main.py new -d ./data/medium -e 400 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()" --message 64 --name unet-combined-noise-64bit  &
sleep 1
tail -f nohup.out
