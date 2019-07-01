mkdir -p data
cd ./data/
if [! -f ./test2017.zip ]; then
  echo "Downloading test2017.zip  dataset from coco..."
  wget http://images.cocodataset.org/zips/test2017.zip
else
  echo "test2017.zip already exists, skipping download"
fi
echo "Download finished"

echo "Unzipping files into test2017/ ..."
unzip -q test2017.zip
echo "Done"

echo "Creating small dataset..."
mkdir -p small/train/train_class
mkdir -p small/val/val_class

find ./test2017 -maxdepth 1 -type f -print0 | head -z -n 200 | xargs -0 -r -- cp -t "small/train/train_class" --
find ./small/train/train_class -maxdepth 1 -type f -print0 | head -z -n -100 | xargs -0 -r -- mv -t "small/val/val_class"
echo "Done"
echo "./small/train/ $(ls -l ./small/train/train_class | egrep -c '^-') files"
echo "./small/val/  $(ls -l ./small/val/val_class   | egrep -c '^-') files"


echo "Creating medium dataset"
mkdir -p medium/train/train_class
mkdir -p medium/val/val_class

find ./test2017 -maxdepth 1 -type f -print0 | head -z -n 11000 | xargs -0 -r -- cp -t "medium/train/train_class" --
find ./medium/train/train_class -maxdepth 1 -type f -print0 | head -z -n 1000 | xargs -0 -r -- mv -t "medium/val/val_class"
echo "Done"
echo "./medium/train/ $(ls -l ./medium/train/train_class | egrep -c '^-') files"
echo "./medium/val/  $(ls -l ./medium/val/val_class   | egrep -c '^-') files"

echo "Removing test2017 folder..." 
rm -rf ./test2017
echo "Done"

