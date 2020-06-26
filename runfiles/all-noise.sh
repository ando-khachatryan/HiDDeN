nohup python -u src/main.py unet -e 300 --noise "crop(0.4,0.55)+cropout(0.25,0.35)+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()+quant()" &
sleep 1
tail -f nohup.out

sleep 300 

nohup python -u src/main.py hidden -e 300 --noise "crop(0.4,0.55)+cropout(0.25,0.35)+dropout(0.25,0.35)+resize(0.4,0.6)+jpeg()+quant()" &
sleep 1
tail -f nohup.out