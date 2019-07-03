nohup python -u ./src/main.py new -d ./data/medium -e 200 --noise 'quant()' --name quantization &
sleep 1
tail -f nohup.out
