import glob
from PIL import Image
import numpy as np



class ImageNetData:


    def __init__(self, config):

        self.lastIndex = 0

        images = glob.glob(config['imagenet_location'] + "*")

        for image in images:
            assert "jpg" in image

        self.numExamples = len(images)
        self.images = images

        self.mb_size = config['mb_size']

    def normalize(self, x):
        return (x / 127.5) - 1.0

    def denormalize(self, x):
        return (x + 1.0) * 127.5

    def getBatch(self):
        
        imageLst = []

        imageLst_4 = []
        imageLst_8 = []
        imageLst_16 = []
        imageLst_32 = []

        index = self.lastIndex

        while len(imageLst) < self.mb_size:
            image = self.images[index]
            imgObj = Image.open(image)

            imgObj = imgObj.resize((256,256))
            img = np.asarray(imgObj)
            if img.shape == (256,256,3):
                imageLst.append([img])


                imgObj = imgObj.resize((4,4))

                img = np.asarray(imgObj)

                imageLst_4.append([img])

                imgObj = imgObj.resize((8,8))

                img = np.asarray(imgObj)

                imageLst_8.append([img])
                imgObj = imgObj.resize((16,16))

                img = np.asarray(imgObj)

                imageLst_16.append([img])
                imgObj = imgObj.resize((32,32))

                img = np.asarray(imgObj)

                imageLst_32.append([img])


            index += 1
            if index >= self.numExamples:
                index = 0


            imgObj.close()

        x = np.vstack(imageLst).astype('float32')
        x_4 = np.vstack(imageLst_4).astype('float32')
        x_8 = np.vstack(imageLst_8).astype('float32')
        x_16 = np.vstack(imageLst_16).astype('float32')
        x_32 = np.vstack(imageLst_32).astype('float32')

        self.lastIndex = index + 1

        return {'x' : x, 'x_4' : x_4, 'x_8' : x_8, 'x_16' : x_16, 'x_32' : x_32}

if __name__ == "__main__":

    config = {}
    
    config["imagenet_location"] = "/u/lambalex/data/imagenet/"


    imageNetData = ImageNetData(config)

    for i in range(0,100000):
        x = imageNetData.getBatch()
        #print (x - imageNetData.denormalize(imageNetData.normalize(x))).mean()
        print imageNetData.normalize(x).max()
        print imageNetData.normalize(x).min()



