from sparrow.strategy.abstract_model import AbstractModel


class LRModel(AbstractModel):
    def __init__(self):
        pass

    def run(self,args):
        print("LR",args)
