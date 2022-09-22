import sys
import time
from configs.config_generator import gen_config
from run import run_exp
from logs.clear_logs import clear_all


def gen_grid():
    SE_div = [[i, i, i] for i in [1, 2, 3, 4, 5, 6]]
    filters = [[i, i + i // 2, i + i] for i in range(128, 192, 16)]
    opts = []
    for i in SE_div:
        for j in filters:
            opts.append(dict(SE_div=i, filters=j))
    print(f"Generated grid group size: {len(opts)}")
    print(
        f"--------------\n"
        f"INFO: The excepted experiment time is no longer than {20 * len(opts)} minutes, or {20 * len(opts) / 60} hours"
        f"\n--------------\n"
    )
    return opts


def optimize(dataset):
    opts = gen_grid()

    best_val_acc = -1
    best_model_config = None

    overallTimeS = time.time()

    for i in range(len(opts)):
        cfg = gen_config("configs/" + dataset + ".json", opts[i])["config"]
        best_accuracy = run_exp(dataset, cfg, save_details=False)
        if best_accuracy > best_val_acc:
            best_val_acc = best_accuracy
            best_model_config = opts[i]

    overallTimeE = time.time()

    print(
        f"\nINFO: optimize process terminated after {overallTimeS - overallTimeE} seconds."
    )

    print(f"the best parameter combination is : {best_model_config}")
    print(f"the best eval accuracy is {best_val_acc}")
    return


def main(argv):
    if argv[0] == "cifar10":
        print("running gdbls on cifar10 dataset...")
    clear_all()
    optimize(argv[0])


if __name__ == "__main__":
    main(sys.argv[1:])
