nohup python -u ./src/main.py hidden -e 400 -b 32 --noise "crop((0.4,0.55),(0.4,0.55))+cropout((0.25,0.35),(0.25,0.35))+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()+quant()" --name hidden-combined-noise > hidden-combined-noise.out &
sleep 1
tail -f hidden-combined-noise.out
