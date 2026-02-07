mkdir coco
cd coco
mkdir images
cd images
mkdir mscoco
cd mscoco
mkdir images
cd images

wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
rm train2014.zip

wget http://images.cocodataset.org/zips/val2014.zip
unzip val2014.zip
rm val2014.zip

wget http://images.cocodataset.org/zips/test2014.zip
unzip test2014.zip
rm test2014.zip

cd ../../../

wget http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

unzip refcoco.zip

rm refcoco.zip