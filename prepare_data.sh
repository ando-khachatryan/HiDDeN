mkdir -p data
cd ./data/

download_and_unpack() {
ZIP_FILE=$1
UNZIP_FOLDER=$2
DS_NAME=$3
DS_TRAIN_SIZE=$4
DS_VAL_SIZE=$5

if [ ! -f ./$ZIP_FILE ]; then
  echo "Downloading $ZIP_FILE  dataset from coco..."
  wget http://images.cocodataset.org/zips/$ZIP_FILE.zip
  echo "Download finished"
else
  echo "$ZIP_FILE already exists, skipping download"
fi
if [ ! -d ./$UNZIP_FOLDER ]; then
  echo "Unzipping files into $UNZIP_FOLDER ..."
  unzip -q $ZIP_FILE
  echo "Done"
else
  echo "Unzipped folder $UNZIP_FOLDER already exists"
fi
echo "Creating $DS_NAME dataset..."
mkdir -p $DS_NAME/train/train_class
mkdir -p $DS_NAME/val/val_class

find ./$UNZIP_FOLDER -maxdepth 1 -type f -print0 | head -z -n $(($DS_TRAIN_SIZE+$DS_VAL_SIZE)) | xargs -0 -r -- cp -t "$DS_NAME/train/train_class" --
find ./$DS_NAME/train/train_class -maxdepth 1 -type f -print0 | head -z -n $DS_VAL_SIZE | xargs -0 -r -- mv -t "$DS_NAME/val/val_class"
echo "Done"
echo "./$DS_NAME/train/ $(ls -l ./$DS_NAME/train/train_class | egrep -c '^-') files"
echo "./$DS_NAME/val/  $(ls -l ./$DS_NAME/val/val_class   | egrep -c '^-') files"
}
# ZIP_FILE='test2017.zip'
# UNZIP_FOLDER='test2017'
# DS_NAME='small'
# DS_TRAIN_SIZE=100
# DS_VAL_SIZE=100
download_and_unpack 'test2017.zip' 'test2017' 'small'   100    100
download_and_unpack 'test2017.zip' 'test2017' 'medium'  10000  1000
download_and_unpack 'train2017.zip' 'train2017' 'large' 100000 3000
