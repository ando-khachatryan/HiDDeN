nohup python -u ./src/unet_main.py new -d ./data/medium -e 300 --noise "crop((0.2,0.25),(0.2,0.25))" --enc-loss-weight 5 --adv-loss-weight 0.005 --name unet-crop &
sleep 1
tail -f nohup.out
