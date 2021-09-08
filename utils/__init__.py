import yaml, inspect, importlib
from collections.abc import Mapping
from datetime import datetime

def to_pair(data):
	r"""Converts a single or a tuple of data into a pair. If the data is a tuple with more than two elements, it selects
	the first two of them. In case of single data, it duplicates that data into a pair.
	Args:
		data (object or tuple): The input data.
	Returns:
		Tuple: A pair of data.
	"""
	if isinstance(data, tuple):
		return data[0:2]
	return (data, data)

def update_dict(orig_dict, new_dict):
    for key, val in new_dict.items():
        if isinstance(val, Mapping):
            tmp = update_dict(orig_dict.get(key, { }), val)
            orig_dict[key] = tmp
        elif isinstance(val, list):
            orig_dict[key] = (orig_dict.get(key, []) + val)
        else:
            orig_dict[key] = new_dict[key]
    return orig_dict

def read_yaml(file):
	with open(file, 'r') as f:
		data = yaml.load(f, yaml.FullLoader)
	return data

def get_module_classes(module):
    return [m[1] for m in inspect.getmembers(module, inspect.isclass) \
        if m[1].__module__ == module.__name__]

def count_layer_instance(net, instance):
    c = 0
    for i, j in net.named_modules():
        if isinstance(j, instance):
            c += 1
    return c

def write_perf_file(network, task, data_len, model_type, kernels, epochs, train_perf, test_perf):
    with open('performance_{}.txt'.format(task), 'a') as f:
        f.write("""

            ######## {} PERFORMANCES ########

            Date : {}
            Network : {}
            Data length : {}
            Type : {}
            Filters : {}
            Number of Kernels : {}
            Epochs : {}
            Train performances (correct, incorrect + silence, silence) : {}
            Test performances (correct, incorrect + silence, silence) : {}
            
            ######## END ########

            """.format(task.upper(),
                datetime.now(), network.name, data_len, model_type, kernels[0].__class__.__name__, len(kernels),
                str(epochs),
                str(train_perf),
                str(test_perf)
        ))
    return