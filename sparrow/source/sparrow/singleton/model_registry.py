class ModelRegistry(object):
    def __init__(self):
        self.dict = {}

    def register(self, model):
        self.dict[type(model).__name__] = model

    def get(self, name):
        return self.dict[name]


model_registry = ModelRegistry()
