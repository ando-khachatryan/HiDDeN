nohup python -u ./src/unet_main.py new -d ./data/medium -e 200  --enc-loss-weight 5 --adv-loss-weight 0.005 --name unet-no-noise &
sleep 1
tail -f nohup.out
