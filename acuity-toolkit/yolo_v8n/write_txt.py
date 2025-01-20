#############将量化图片写入txt文件中

import os

image_path = "./data/"
txt_path = "dataset.txt"

f = open(txt_path, "w")

for i in os.listdir(image_path):
    if i.split(".")[-1] == "jpg":
        f.write(image_path + i + "\n")

f.close()
