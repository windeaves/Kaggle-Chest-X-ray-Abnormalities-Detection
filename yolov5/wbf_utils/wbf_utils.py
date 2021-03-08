import torch
from torchvision.transforms import functional as F


class TransformOneModel:
    def __init__(self, rotate=90.0, prob=0.5):
        self.angle = rotate
        self.prob = prob

    def rotate(self, x):
        return F.rotate(x, self.angle)

    def hflip(self, x):
        return F.hflip(x)

    def vflip(self, x):
        return F.vflip(x)

    def __call__(self, img):
        img1 = self.rotate(img)
        img2 = self.vflip(img)
        img3 = self.hflip(img)

        return torch.cat((img, img1, img2, img3), 0)


def wbf(pred):
    """
    Args:
        pred: detections with shape: bxnx6 (x1, y1, x2, y2, conf, cls)

    Returns:
        detections with shape: nx6 (x1, y1, x2, y2, conf, cls) after wbf
    """

