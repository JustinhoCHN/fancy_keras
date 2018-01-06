import glob
import cv2
import math
from PIL import Image
import numpy as np
from keras.applications import ResNet50
from keras.utils import Sequence
from keras import optimizers


def read_img(path, target_size):
    try:
        img = Image.open(path).convert("RGB")
        img_rs = img.resize(target_size)
    except Exception as e:
        print(e)
    else:
        x = np.expand_dims(np.array(img_rs), axis=0)
        return x

def my_gen(path, batch_size, target_size):
    img_list = glob.glob(path + '*.jpg')    # 获取path里面所有图片的路径
    steps = math.ceil(len(img_list) / batch_size)
    print("Found %s images."%len(img_list))
    while True:
        for i in range(steps):
            batch_list = img_list[i * batch_size : i * batch_size + batch_size]
            x = [read_img(file, target_size) for file in batch_list]
            batch_x = np.concatenate([array for array in x])
            y = np.zeros((batch_size, 1000))    # 你可以读取你写好的标签，这里为了演示简洁就全设成0
            yield batch_x, y    # 把制作好的x, y生成出来
			

class SequenceData(Sequence):
    def __init__(self, path, batch_size, target_size):
        # path：存放所有图片的文件夹
        self.path = path
        self.batch_size = batch_size
        self.target_size = target_size
        self.x_filenames = glob.glob(self.path + '*.jpg')
        self.x_filenames.sort(key=lambda x: int(x.split('/')[-1][:-4]))

    def __len__(self):
        # 让代码知道这个序列的长度
        num_imgs = len(glob.glob(self.path + '*.jpg'))
        return math.ceil(num_imgs / self.batch_size)

    def __getitem__(self, idx):
        # 迭代器部分
        batch_x = self.x_filenames[idx * self.batch_size: (idx + 1) * self.batch_size]
        x_arrays = np.array([self.read_img(filename) for filename in batch_x])    # 读取一批图片
        batch_y = np.zeros((self.batch_size, 1000))    # 为演示简洁全部假设为0

        return x_arrays, batch_y

    def read_img(self, x):
        try:
            img = cv2.imread(x)    # 这里用cv2是因为读取图片比pillow快
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    #opencv读取通道顺序为BGR，所以要转换
            img = cv2.resize(img, self.target_size)
        except Exception as e:
            print(e)
        else:
            return img
			
if __name__ = '__main__':
	path = '/img/'
	model = ResNet50()
	model.compile(optimizer=optimizers.Adam(1e-4), loss='categorical_crossentropy')

	batch_size = 64
	steps = math.ceil(len(glob.glob(path + '*.jpg')) / batch_size)
	target_size = (224, 224)
	data_gen = my_gen(path, batch_size, target_size)    # 使用上面写好的generator
	# 或者使用下面的Sequence数据
	# sequence_data = SequenceData(path, batch_size, target_size)

	loss = model.fit_generator(data_gen, steps_per_epoch=steps, epochs=10, verbose=1)

	# 也可以使用下面的多进程
	# loss = model.fit_generator(sequence_data, steps_per_epoch=steps, epochs=10, verbose=1, use_multiprocessing=True, workers=2)