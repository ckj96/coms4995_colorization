import os
cnt_num = 0
cnt_cat = 0
root_path = ('/home/cc4192/data/data/vision/torralba/deeplearning/images256')
dir1 = os.listdir(root_path)
for directory in dir1:
    dir_path = root_path + '/' + directory
    if os.path.isdir(dir_path):

        print(directory, len(os.listdir(dir_path)))
        print(os.listdir(dir_path))
        cnt_cat += 1
        cnt_num += len(os.listdir(dir_path))
print(cnt_cat, cnt_num)
