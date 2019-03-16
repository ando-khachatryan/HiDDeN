nohup python -u main.py new -d /data/coco/10K -e 200 -b 32 --noise 'crop((0.4,0.55),(0.4,0.55))+quant()' --name crop_quantization & 
sleep 1
tail -f nohup.out
