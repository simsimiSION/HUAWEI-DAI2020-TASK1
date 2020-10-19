import yaml

class Dict(dict):
    __setattr__ = dict.__setitem__
    __getattr__ = dict.__getitem__

def dict2obj(dictObj):
    if not isinstance(dictObj, dict):
        return dictObj
    d = Dict()
    for k, v in dictObj.items():
        d[k] = dict2obj(v)
    return d

def get_args():
    """返回参数
    """
    f = open('/home/HUAWEI-DAI2020/task1/starter_kit/train_example/utils/params.yaml', 'r')
    file_data = f.read()
    f.close()

    return dict2obj(yaml.load(file_data))