# 必要なものをインポート
import chainer
from chainer import training, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer.training import extensions

import numpy as np
import sys

from model import NN, CNN

# 各パラメータの定義
# バッチサイズは、データをまとめる数。
# エポックは、学習回数を表す。
batch_size = 100
epoch_size = 3
modelname = "mnist_NN.h5"

# 学習モデルの定義
model = NN()
if (len(sys.argv) > 1) and (sys.argv[1] == "CNN"):
	model = CNN()
	modelname = "mnist_CNN.h5"

# データセットの読み込み。
# 今回は、MNIST(エムニスト)という手書き文字データセットを使用。
# 学習データとテストデータで分ける。
# train, testのそれぞれには、１次元配列の画像情報(list)と、
# 画像の答え(int)の２つの情報が入っている。
# image size = 784 -> (1 x 28 x 28)の３次元に指定。
train, test = chainer.datasets.get_mnist(ndim=3)

# 学習データのイテレーションを作成
# バッチサイズ分のデータに区切る。
iterator = chainer.iterators.SerialIterator(train, batch_size)

# Adamは最適化手法。逆伝播のやり方を決める。
optimizer = chainer.optimizers.Adam()
# 学習モデルのセットアップ。
optimizer.setup(model)

# 学習の準備
# アップデータは、iteratorを順伝播させ、
# optimizerを用いて逆伝播させる。
updater = training.StandardUpdater(iterator, optimizer)
# トレーナーは、何回学習を繰り返すのかを決める。
# ここではepoch_size分、学習を繰り返す。
trainer = training.Trainer(updater, (epoch_size, 'epoch'))

# 進行状況の表示をする。
trainer.extend(extensions.ProgressBar())

# 学習開始
trainer.run()

# 学習後のモデルの正当性を確認
print("Answer:", end="")

ok = 0
# テストデータ全てを試す
for test_i in test:
	# test_iの画像情報をValiable(Chainer専用の型)に変換
	x = Variable(np.array([test_i[0]], dtype=np.float32))
	# test_iの正解番号
	t = test_i[1]
	# モデルにxを入力し、順伝播させた結果を取得(list)
	out = model.fwd(x)
	# 出力が大きいユニットの番号を回答とする。
	ans = np.argmax(out.data)
	if(ans == t): ok += 1

print((ok*1.0) / len(test))

# モデルをh5形式で保存
serializers.save_hdf5(modelname, model)

