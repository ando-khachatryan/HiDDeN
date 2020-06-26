# python -m wm unet-conv --noise 'blur(2)+crop(0.28,0.32)+cropout(0.28,0.32)+dropout(0.28,0.32)+jpeg()+resize(0.28,0.32)' --epochs 400
# sleep 300
# python -m wm unet-attn --noise 'blur(2)+crop(0.28,0.32)+cropout(0.28,0.32)+dropout(0.28,0.32)+jpeg()+resize(0.28,0.32)' --epochs 400
# python -m wm unet-attn --noise 'crop(0.28,0.32)' --epochs 200
# sleep 300
# python -m wm hidden --noise 'crop(0.28,0.32)' --epochs 200

