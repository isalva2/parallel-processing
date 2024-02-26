import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    data = pd.read_csv("./data/data1.csv", header=0)
    data["Threads"] = data["Threads"].astype(str)
    return data


def draw_figure1():
    data = load_data()
    
    fig, ax1 = plt.subplots()
    
    for key in data.keys():
        if key != "Threads":
            ax1.plot(
                data["Threads"],
                data[key],
                label=f"N={key}",
                marker="x"
            )
    
    ax1.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Workload")
    ax1.set_yscale("log", base = 10)
    ax1.set_ylabel("Runtime (log (ms))")
    ax1.set_xlabel("Threads")
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    plt.savefig("./figures/experiment1.png", dpi = 200, bbox_inches="tight")

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
                marker = "x",
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
    ax1.set_ylim(0, 20)
    ax1.set_xlim(1,129)
    custom_ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    ax1.set_xticks(custom_ticks)
    ax1.set_ylabel("Speedup")
    ax1.set_xlabel("Threads")
    plt.savefig("./figures/experiment1-sp.png", dpi = 200, bbox_inches="tight")
    # plt.show()

def draw_figure3():
    data = load_data()
    fig, ax1 = plt.subplots()
    data["Threads"] = data["Threads"].astype(int)
    
    markers = ["o", "^", "s", "+", "*", "x", "d"]
    
    for i, key in enumerate(data.keys()):
        if key != "Threads":
            ts = data[key].iloc[0]
            sp = f"sp_{key}"
            data[sp] = ts/data[key]
            
            ax1.plot(
                data["Threads"],
                data[sp],
                label = f"Initial N={key}",
                color = "gray",
                marker = markers[i-1],
                markersize = 3.0,
                linewidth = 0.5,
                alpha = 0.5
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
    
    data2 = pd.read_csv("./data/data2.csv", header=0)
    
    data2["sp_2000"] = data2["2000"].iloc[0]/data2["2000"]
    data2["sp_5000"] = data2["5000"].iloc[0]/data2["5000"]
    
    ax1.plot(
        data2["Threads"],
        data2["sp_2000"],
        marker = "x",
        label = "Optimized N=2000"
    )
    
    ax1.plot(
        data2["Threads"],
        data2["sp_5000"],
        marker = "x",
        label = "Optimized N=5000"
    )
    
    
    ax1.legend()
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Workload") 
    ax1.set_ylim(0,25)
    ax1.set_xlim(1,129)
    custom_ticks = [1, 4, 8, 16, 32, 64, 128]
    ax1.set_xticks(custom_ticks)
    ax1.set_ylabel("Speedup")
    ax1.set_xlabel("Threads")
    plt.savefig("./figures/experiment2.png", dpi = 200, bbox_inches="tight")
    print(data2)
    # plt.show()
            
if __name__ == "__main__":
    draw_figure1()
    draw_figure2()
    draw_figure3()
