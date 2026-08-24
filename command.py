# scp -r "image_30k" trinh.minh.hieu@10.0.4.239:/home/trinh.minh.hieu/HiDDeN/data

# py main.py new --name test --data-dir data --batch-size 32


"""Example usage:
        - Crop 70-90% each dimension
        - Cropout 70-90% each dimension
        - Retain 80-95% of pixels, drop the rest
        - Downsample to 50-90% of original size
        - JPEG compression
"""
# python main.py new --name test_6 --data-dir ./image_30k --batch-size 30 --noise "crop+cropout+dropout+resize+color_jitter+color_grading+sharpness+identity"

# python main.py new --name test_8 --data-dir ./image_30k --batch-size 30 --noise "crop+cropout+dropout" --continue-from-folder "HiDDeN/runs/test_6 2026.08.17--10-36-24"

# +jpeg()


# export CUDA_VISIBLE_DEVICES=1


# python test_model.py \
#   -o "runs/test_6 2026.08.17--10-36-24/options-and-config.pickle" \
#   -c "runs/test_6 2026.08.17--10-36-24/checkpoints/test_6--epoch-300.pyt" \
#   -s "./data/test/test_class/03.jpg"


# "Combined([Crop(0.8, 1.0), Cropout(0.05, 0.1), Dropout(0.1), Identity()])"