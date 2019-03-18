# 必要なものをインポート
import chainer
from chainer import training, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.training import extensions

import numpy as np
import sys
from PIL import Image, ImageOps

from model import NN, CNN

# 各パラメータ
model = NN()
image_name = ""
CNN_flag = False
Invert = False

# コマンドライン引数を設定
def set_args():
	global CNN_flag, image_name, Invert

	if (len(sys.argv) > 1):
		if(sys.argv[1] == "CNN"):
			CNN_flag = True

	if (len(sys.argv) > 2):
		image_name = sys.argv[2]

	if (len(sys.argv) > 3):
		Invert = True

# 画像を読み込み、モデルが読み込めるように整形する関数
def load_image(image_name, CNN_flag=False, Invert=False):
	im = Image.open(image_name).convert('L')
	if(Invert == True): im = ImageOps.invert(im)
	im = im.resize((28,28))
	# im.save("test.png")
	im = np.array(im, dtype=np.float32)
	if(CNN_flag == True):im = im.reshape((1,1,28,28))
	else: im = im.reshape((1,28,28))
	im = (255 - im) / 255
	# print(im)

	x = Variable(im)

	return x

# 学習モデルの読み込み
def load_model(mname=False):
	global model, CNN_flag, image_name, Invert

	if(mname == True):
		model = CNN()
		serializers.load_hdf5("mnist_CNN.h5", model)
		CNN_flag = True

	else:
		serializers.load_hdf5("mnist_NN.h5", model)

def classifier(x):
	global model
	# 分類
	# モデルにxを入力し、順伝播させた結果を取得
	# 今回は10回やった中で、一番良かったものを抽出する。
	# print("Classified:", end="")
	ans_list = np.zeros(10)

	for _ in range(10):
		out = model.fwd(x)
		ans_list[ np.argmax(out.data) ] += 1
	# 出力が大きいユニットの番号を回答とする。
	# print(ans_list)
	ans = np.argmax(ans_list)

	# print("Answer:",ans)
	return ans

if __name__ == "__main__":
	set_args()
	load_model(CNN_flag)
	x = load_image(image_name, CNN_flag, Invert)
	ans = classifier(x)
	print("Answer:", ans)

