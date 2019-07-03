nohup python -u ./src/main.py new -d ./data/medium -e 200 --name no-noise &
sleep 1
tail -f nohup.out