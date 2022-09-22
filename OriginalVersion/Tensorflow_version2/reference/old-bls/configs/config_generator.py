from configs.config_reader import read_config


def get_keys(target):
    keys = []
    for key in target.keys():
        keys.append(key)
    for val in target.values():
        if isinstance(val, dict):
            keys.extend(get_keys(val))
    return keys


def edit_config(k, val, target):
    valid = False
    if k in target.keys():
        target[k] = val
        print(f"successfully accessed target key-value {k}-{val}")
        valid = True
    else:
        for value in target.values():
            if isinstance(value, dict):
                valid = valid | edit_config(k, val, value)
    return valid


def gen_config(config_template, options=None):
    """
    this is a simple config generator to optimize the parameter a given model.
    :param config_template: config_template path of a certain dataset.
    :param options: dictionary of parameter entries.
    :return: the generated config dictionary.
    """
    target = read_config(config_template)
    keys = get_keys(target)
    if options is not None:
        for key, item in options.items():
            if key not in keys:
                print(f"ERROR: wrong parameter name :[{key}].")
                exit(-1)
            else:
                if not edit_config(key, item, target):
                    print(
                        f"ERROR: failed to edit the original config file for key-val :{key}-{item}"
                    )
                else:
                    print(
                        f"INFO : new config dict generated, currently customed option size: {len(options)}"
                    )

    return target
