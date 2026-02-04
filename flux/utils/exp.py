import os
import secrets


def setup_exp_idx(output_dir):
    exp_idx = 0
    for folder in os.listdir(output_dir):
        # noinspection PyBroadException
        try:
            curr_exp_idx = max(exp_idx, int(folder.split("-")[0].lstrip("0")))
            exp_idx = max(exp_idx, curr_exp_idx)
        except:
            pass
    return exp_idx


def setup_exp_name(output_dir, train_data_dir):
    exp_idx = setup_exp_idx(output_dir)
    exp_name = "{0:0>5d}-{1}-{2}".format(
        exp_idx + 1,
        secrets.token_hex(2),
        os.path.basename(os.path.normpath(train_data_dir)),
    )
    return exp_name

