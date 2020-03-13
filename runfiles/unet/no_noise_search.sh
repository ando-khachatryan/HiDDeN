python ./src/unet_main.py new -d ./data/medium -e 200  --enc-loss-weight 7.0 --adv-loss-weight 0.001 --name unet-no-noise-7.0-0.001 > unet-no-noise-search.log
sleep 300
python ./src/unet_main.py new -d ./data/medium -e 200  --enc-loss-weight 5.0 --adv-loss-weight 0.001 --name unet-no-noise-4.0-0.001 > unet-no-noise-search.log
sleep 300
python ./src/unet_main.py new -d ./data/medium -e 200  --enc-loss-weight 5.0 --adv-loss-weight 0.002 --name unet-no-noise-5.0-0.002 > unet-no-noise-search.log
sleep 300
python ./src/unet_main.py new -d ./data/medium -e 200  --enc-loss-weight 3.0 --adv-loss-weight 0.005 --name unet-no-noise-3.0-0.005 > unet-no-noise-search.log

