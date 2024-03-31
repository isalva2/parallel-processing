# libraries
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("mpi-results-1.csv", header = 0)
    for col in df.columns:
        if col != "proc":
            df[col] = df[col].astype(float)
    return df

def draw_figure_1():
    df = load_data()
    df = df.iloc[1:]
    
    fig, ax = plt.subplots(figsize = (10, 6))
    
    cum_IO = df["t_IO"]
    cum_sched = df["t_IO"] + df["t_sched"]
    cum_calc_total = cum_sched + df["t_calc"]
    
    ax.fill_between(df["proc"], 0, cum_IO, alpha = 0.55, label = "I/O")
    ax.fill_between(df["proc"], cum_IO, cum_sched, alpha = 0.55, label = "Scheduling")
    ax.fill_between(df["proc"], cum_sched, cum_calc_total, alpha = 0.55, label = "Calculations")
    ax.plot(df["proc"], cum_IO, marker = "o", zorder = 1)
    ax.plot(df["proc"], cum_sched, marker = "o", zorder = 1)
    ax.plot(df["proc"], cum_calc_total, marker = "o", zorder = 1)
    
    ax.legend(title = "Runtime Split", ncols = 3, framealpha = 1,)
    
    ax.set_yscale("log", base = 10)
    ax.set_xlim(2, 160)
    x_ticks = list(df["proc"])
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)
    ax.set_axisbelow(True)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Processes")
    ax.set_ylabel("Runtime log(Seconds)")
    
    plt.savefig("figures/ex-1.png", dpi = 200)
    
def draw_figure_2():
    df = load_data()
    
    ts = df["t_total"][0]
    
    df["sp"] = df["t_total"].apply(
        lambda tp: ts / tp 
    )
    
    fig, ax = plt.subplots(figsize = (10, 6))
    
    plt.plot(df["proc"], df["sp"], marker = "o", label = "Experimental")
    plt.plot((0,160), (0,160), label = "Theoretical")
    
    ax.legend(title="Speedup", framealpha = 1)

    ax.set_xlim(0, 160)
    ax.set_ylim(0, 55)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)
    ax.set_axisbelow(True)
    
    ax.set_xlabel("Processes")
    ax.set_ylabel("Speedup")
    
    plt.savefig("figures/ex-2.png", dpi = 200)
    
def draw_figure_3():
    df = load_data()
    
    ts = df["t_total"][0]
    
    df["sp"] = df["t_total"].apply(
        lambda tp: ts / tp 
    )
    
    fig, ax = plt.subplots(figsize = (6, 6))
    
    plt.plot(df["proc"], df["sp"], marker = "o", label = "Experimental")
    plt.plot((0,160), (0,160), label = "Theoretical", zorder = 1)
    
    ax.legend(title="Speedup", framealpha = 1)
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 40)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)
    ax.set_axisbelow(True)
    ax.title.set_text("Speedup Detail")
    
    ax.set_xlabel("Processes")
    ax.set_ylabel("Speedup")

    
    plt.savefig("figures/ex-2-detail.png", dpi = 200)

if __name__ == "__main__":
    draw_figure_1()
    draw_figure_2()
    draw_figure_3()