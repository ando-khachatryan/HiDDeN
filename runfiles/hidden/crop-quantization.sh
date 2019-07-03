nohup python -u ./src/main.py new -d ./data/medium -e 200 --noise 'crop((0.4,0.55),(0.4,0.55))+quant()' --name crop_quantization &
sleep 1
tail -f nohup.out