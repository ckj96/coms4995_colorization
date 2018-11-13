from pycocotools.coco import COCO
import numpy as np
import os
import shutil

def copy(src, dst):
    if os.path.islink(src):
        linkto = os.readlink(src)
        os.symlink(linkto, dst)
    else:
        shutil.copy(src, dst)


dataDir = '/home/cc4192/recom/image-similarity-using-deep-ranking'
dataType = 'val2017'
annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

coco = COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms = [cat['name'] for cat in cats]

for catNm in nms:
    print('processing category:', catNm)
    catIds = coco.getCatIds(catNms=catNm)
    imgIds = coco.getImgIds(catIds=catIds)
    Imgs = coco.loadImgs(imgIds)

    dirNm = os.path.join(dataDir, 'val', catNm)
    if not os.path.isdir(dirNm):
        os.mkdir(dirNm)

    for img in Imgs:

        name = img['file_name']
        img_path = os.path.join(dataDir, dataType, name)
        des_path = os.path.join(dirNm, name)
        copy(img_path, des_path)

