import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv("./data/data1.csv", header=0)
    data["Threads"] = data["Threads"].astype(str)
    return data


def draw_figure2():
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
    ax1.set_yscale("log", base = 10)
    ax1.set_ylabel("Runtime (log (ms))")
    ax1.set_xlabel("Threads")
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    plt.savefig("./figures/experiment1.png", dpi = 200, bbox_inches="tight")
    plt.show()

def draw_figure2():
    data = load_data()
    fig, ax1 = plt.subplots()
    data["Threads"] = data["Threads"].astype(int)
    
    for key in data.keys():
        if key != "Threads":
            ts = data[key].iloc[0]
            sp = f"sp_{key}"
            data[sp] = ts/data[key]
            
            ax1.plot(
                data["Threads"],
                data[sp],
                label = f"N={key}",
                marker = "o",
                markersize = 3.0,
                linewidth = 1
            )
    
    ax1.plot(
        [0,128],
        [0,128],
        label="Theoretical",
        linestyle = "--",
        linewidth = 1,
        alpha = 0.8,
        color = "red"
    )
    
    ax1.legend()
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Workload") 
    ax1.set_ylim(0,14)
    ax1.set_xlim(1,65)
    custom_ticks = [0, 1, 2, 4, 8, 16, 32, 64]
    ax1.set_xticks(custom_ticks)
    ax1.set_ylabel("Speedup")
    ax1.set_xlabel("Threads")
    plt.savefig("./figures/experiment1-sp.png", dpi = 200, bbox_inches="tight")
    print(data)
    
    # plt.show()
            
if __name__ == "__main__":
    draw_figure2()
