IMG_NAME=test.png
INPUT_IMG=./input_images/$IMG_NAME
OUTPUT_IMG=./results/$IMG_NAME
#python ./colorize.py -img_in $INPUT_IMG -img_out $OUTPUT_IMG
th colorize.lua $INPUT_IMG $OUTPUT_IMG

