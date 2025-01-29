import matplotlib.pyplot as plt


def dict_difference(*dicts):
    sets = [set(d.items()) for d in dicts]
    overlap = sets[0]
    for s in sets[1:]:
        overlap = overlap.intersection(s)
    return [dict(kv.difference(overlap)) for kv in sets]


def unique_keys(this_dict, other_dicts):
    return dict_difference(this_dict, *other_dicts)[0]


def update_dict(d, **new_values):
    new_d = d.copy()
    for k, v in new_values.items():
        new_d[k] = v
    return new_d


def try_plot():
    try:
        plt.show()
    except:
        pass
