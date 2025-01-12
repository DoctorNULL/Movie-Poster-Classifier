import torch
from torch.utils.data import Dataset
from pickle import load
import cv2 as cv

class DataHandler(Dataset):
    def __init__(self, data: str = "train", maxSize = None):
        file = open(data + ".data", 'rb')
        self.data = load(file)
        file.close()

        self.maxSize = maxSize

    def __len__(self):
        if self.maxSize:
            if self.maxSize > len(self.data):
                return len(self.data)
            else:
                return self.maxSize
        return len(self.data)

    def __getitem__(self, item) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data[item]

        img = cv.imread("D:\\MachineLearning\\Datasets\\Multi_Label_dataset\\Images\\" + item.image + ".jpg")
        img = cv.resize(img, (250, 250))

        x = torch.tensor(img, dtype=torch.float)
        x = x / 255
        x = x.permute((2, 0, 1))

        y = torch.tensor(item.labels, dtype= torch.float)

        return x, y

    def getRawImage(self, item):
        item = self.data[item]

        img = cv.imread("D:\\MachineLearning\\Datasets\\Multi_Label_dataset\\Images\\" + item.image + ".jpg")
        cv.imshow(item.image, img)
        cv.waitKey(0)

        cv.destroyAllWindows()