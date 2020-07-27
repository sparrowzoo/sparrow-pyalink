import sys

from sparrow.singleton.model_registry import model_registry
from sparrow.strategy.lr_model import LRModel
from sparrow.strategy.wide_deep_model import WideDeepModel

model_registry.register(LRModel())
model_registry.register(WideDeepModel())


def assemble_arg_dict():
    args = sys.argv[2:]
    arg_dict = {}
    i = 0
    while i < len(args)-1:
        key = args[i]
        i += 1
        val = args[i]
        arg_dict[key] = val
    return arg_dict


#py model_name model_arg1 model_arg2  model_argn
if __name__ == '__main__':
    model_registry.get(sys.argv[1]).run(assemble_arg_dict())
