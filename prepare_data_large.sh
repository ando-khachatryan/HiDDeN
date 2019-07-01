mkdir -p data
cd ./data/
echo "Downloading train2017.zip  dataset from coco..."
wget http://images.cocodataset.org/zips/train2017.zip
echo "Download finished"

echo "Unzipping files into train2017/ ..."
unzip -q train2017.zip
echo "Done"

echo "Creating large  dataset..."
mkdir -p large/train/train_class
mkdir -p large/val/val_class

find ./train2017 -maxdepth 1 -type f -print0 | head -z -n 103000 | xargs -0 -r -- cp -t "large/train/train_class" --
find ./large/train/train_class -maxdepth 1 -type f -print0 | head -z -n -3000 | xargs -0 -r -- mv -t "large/val/val_class"
echo "Done"
echo "./large/train/ $(ls -l ./large/train/train_class | egrep -c '^-') files"
echo "./large/val/  $(ls -l ./large/val/val_class   | egrep -c '^-') files"

echo "Removing train2017 folder..." 
rm -rf ./train2017
echo "Done"

