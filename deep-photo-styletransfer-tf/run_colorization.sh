TEST_IMG=tower.png
#SEGMENTATION_IMG=in8.png
STYLE_IMG=tar7.png
#TEST_PATH=/home/yanbeipang/siggraph2016_colorization/results
TEST_PATH=/home/cc4192/siggraph2016_colorization/input_images
#/home/cc4192/semantic-segmentation-pytorch/results
STYLE_PATH1=/home/cc4192/colorr/deep-photo-styletransfer-tf/examples/input
STYLE_PATH2=/home/cc4192/colorr/deep-photo-styletransfer-tf/examples/style
#TEST_PATH=/home/cc4192/colorr/deep-photo-styletransfer-tf/examples/input/
#TEST_PATH=/home/yanbeipang/segmentation/semantic-segmentation-pytorch/gray_images
#SEGMENTATION_IMG_PATH=/home/yanbeipang/segmentation/semantic-segmentation-pytorch/results
SEGMENTATION_IMG_PATH=/home/cc4192/semantic-segmentation-pytorch/results
#cp $TEST_PATH/$TEST_IMG ./gray_images/
#cp $SEGMENTATION_IMG_PATH/$SEGMENTATION_IMG ./gray_images/segmentation/

python deep_photostyle.py --content_image_path $TEST_PATH/$TEST_IMG --style_image_path $STYLE_PATH2/$STYLE_IMG --content_seg_path $SEGMENTATION_IMG_PATH/$TEST_IMG --style_seg_path $SEGMENTATION_IMG_PATH/$STYLE_IMG --style_option 2
