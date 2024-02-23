import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv("./data/data1.csv", header=0)
    data["Threads"] = data["Threads"].astype(str)
    return data


def draw_figure():
    data = load_data()
    
    fig, ax1 = plt.subplots()
    
    for key in data.keys():
        if key != "Threads":
            ax1.plot(
                data["Threads"],
                data[key],
                label=f"N={key}",
                marker="o"
            )
    
    ax1.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Workload")
    ax1.set_yscale("log", base = 2)
    ax1.set_ylabel("Time (ms)")
    ax1.set_xlabel("Threads")
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    plt.savefig("./figures/experiment1.png", dpi = 200, bbox_inches="tight")
    plt.show()
            
if __name__ == "__main__":
    draw_figure()
