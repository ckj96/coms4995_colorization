
 for file in input/*.png;
 do python ./colorize.py -img_in input/${file##*/} -img_out output/${file##*/};
#do echo ${file##*/};
 done

#IMG=building
#python ./colorize.py -img_in input/$IMG.png -img_out output/$IMG.png
