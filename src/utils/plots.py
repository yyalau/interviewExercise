import matplotlib.pyplot as plt
import numpy as np

def save_result(title,  gt, dcbo, path=None):
    nT = len(gt)
    nTrials = len(dcbo[0])
    
    fig, axs = plt.subplots(nT, 1, figsize=(15, 8))
    # fig.tight_layout()
    fig.suptitle(title)
    
    x = np.arange(nTrials)
    for t in range(nT):
        axs[t].plot(x, np.repeat(gt[t], nTrials), label="GT")
        axs[t].plot(x, dcbo[t], label="DCBO")
        axs[t].set_title(f"Time {t}")
        axs[t].legend()
        axs[t].set_xlabel("Trials")
        axs[t].set_ylabel("Value")
        
    if path:
        fig.savefig(path)
        print("Saved image to ", path + ".png")
        
        with open(path + '.txt', "w") as f:
            f.write(",".join([str(float(min(dcbo[t]))) for t in range(nT)]))
            f.write("\n")
            f.write(",".join([str(gt[t]) for t in range(nT)]))


def save_result_bf(title,  pdc, ig, path=None):
    figs, ax = plt.subplots(figsize = (5, 2.5))
    x = np.arange(n_trials:=len(pdc))
    ax.plot(x, pdc, label="P_DC")
    ax.plot(x, ig, label="InfoGain")
    ax.set_xlabel("Trials")
    ax.set_ylabel("Criterion Value")
    ax.legend()
    
    figs.suptitle(title)
    figs.savefig(path + ".png")
    print("Saved image to ", path + ".png")
    
    with open(path + '.txt', "w") as f:
        f.write(",".join([str(pdc[t]) for t in range(n_trials)]))
        f.write("\n")
        f.write(",".join([str(ig[t]) for t in range(n_trials)]))
    