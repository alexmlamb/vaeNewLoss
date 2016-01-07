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

    def normalize(self, x):
        return (x / 127.5) - 1.0

    def denormalize(self, x):
        return (x + 1.0) * 127.5

    def getBatch(self):
        
        imageLst = []

        index = self.lastIndex

        while len(imageLst) < 100:
            image = self.images[index]
            imgObj = Image.open(image)
            imgObj = imgObj.resize((256,256))
            img = np.asarray(imgObj)
            imgObj.close()
            if img.shape == (256,256,3):
                imageLst.append([img])
            index += 1
            if index >= self.numExamples:
                index = 0

        x = np.vstack(imageLst).astype('float32')

        self.lastIndex = index + 1

        return x

if __name__ == "__main__":

    config = {}
    
    config["imagenet_location"] = "/u/lambalex/data/imagenet/"


    imageNetData = ImageNetData(config)

    for i in range(0,100000):
        x = imageNetData.getBatch()
        #print (x - imageNetData.denormalize(imageNetData.normalize(x))).mean()
        print imageNetData.normalize(x).max()
        print imageNetData.normalize(x).min()



