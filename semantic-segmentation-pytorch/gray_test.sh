#TEST_IMG=gray_images/house.jpg
IMG_NAME=in1.png
#TEST_IMG=/home/yanbeipang/siggraph2016_colorization/input_images/$IMG_NAME
#TEST_IMG=/home/cc4192/colorr/deep-photo-styletransfer-tf/examples/style/$IMG_NAME
TEST_IMG=/home/cc4192/colorr/deep-photo-styletransfer-tf/examples/input/$IMG_NAME
MODEL_PATH=baseline-resnet50_dilated8-ppm_bilinear_deepsup
RESULT_PATH=results/

ENCODER=resnet50_dilated16
DECODER=ppm_bilinear_deepsup

#ENCODER=$MODEL_PATH/encoder_epoch_20.pth
#DECODER=$MODEL_PATH/decoder_epoch_20.pth

#
#if [ ! -e $MODEL_PATH ]; then
#  mkdir $MODEL_PATH
#fi
#if [ ! -e $ENCODER ]; then
#  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$ENCODER
#fi
#if [ ! -e $DECODER ]; then
#  wget -P $MODEL_PATH http://sceneparsing.csail.mit.edu/model/pytorch/$DECODER
#fi
#if [ ! -e $TEST_IMG ]; then
#  wget -P $RESULT_PATH http://sceneparsing.csail.mit.edu/data/ADEChallengeData2016/images/validation/$TEST_IMG
#fi

python3 -u test.py \
  --model_path $MODEL_PATH \
  --test_img $TEST_IMG\
  --arch_encoder $ENCODER\
  --arch_decoder $DECODER\
  --fc_dim 2048 \
  --result $RESULT_PATH
