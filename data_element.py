class DataElement(object):
    def __init__(self, image: str, labels:list[int]):
        self.image = image
        self.labels = labels

    def __str__(self):
        return str(self.image) + "\t" + str(self.labels)

    def __repr__(self):
        return str(self)