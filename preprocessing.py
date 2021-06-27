import cv2
import os

# importing function image_processing file
from image_processing import func

path = "signdata/train_img"
path1 = "signdata"
a = ['label']

for i in range(64*64):
    a.append("pixel"+str(i))

label = 0
var = 0
p = 0
q = 0
for (dirpath, dirnames, filenames) in os.walk(path):
    for dirname in dirnames:
        print(dirname)
        for(direcpath, direcnames, files) in os.walk(path+"/" + dirname):
            if not os.path.exists(path1 + "/train_img/" + dirname):
                os.makedirs(path1 + "/train_img/" + dirname)
            if not os.path.exists(path1 + "/test_img/" + dirname):
                os.makedirs(path1 + "/test_img/" + dirname)
            #
            num = 10000000000000
            i = 0
            for file in files:
                var += 1
                actual_path = path+"/"+dirname+"/"+file
                actual_path1 = path1+"/"+"train_img/"+dirname+"/"+file
                actual_path2 = path1+"/"+"test_img/"+dirname+"/"+file
                img = cv2.imread(actual_path, 0)
                bw_image = func(actual_path)
                if i < num:
                    p += 1
                    cv2.imwrite(actual_path1, bw_image)
                else:
                    q += 1
                    cv2.imwrite(actual_path2, bw_image)
                    
                i = i+1
        label = label+1
print(var)
print(p)
print(q)
