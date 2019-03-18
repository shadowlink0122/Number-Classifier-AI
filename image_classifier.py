import numpy as np
import os, sys

import classifier as cf

IGM_DIR = "./images/"
NN = False

# モデルをロード
if (len(sys.argv) > 1):
	if (sys.argv[1] == "CNN"):
		NN = True

print("NN") if NN == False else print("CNN") 
cf.load_model(NN)

for dir_name in os.listdir(IGM_DIR):
	dir_size = 0
	ans_list = np.zeros(10)
	print("DIR:", dir_name, end="")

	for filename in os.listdir(IGM_DIR+dir_name):
		x = cf.load_image(IGM_DIR + dir_name + "/" + filename, NN)
		ans = cf.classifier(x)
		ans_list[ans] += 1
		dir_size += 1

	maxIndex = [i for i, x in enumerate(ans_list) if x == max(ans_list)]
	print("-> Most:", maxIndex)
	print("{0}:".format(dir_size),ans_list)

