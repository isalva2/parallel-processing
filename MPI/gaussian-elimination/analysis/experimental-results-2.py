# libraries
import pandas as pd
import matplotlib.pyplot as plt

def load_data(path):
    df = pd.read_csv(path, header = 0)
    for col in df.columns:
        if col != "proc":
            df[col] = df[col].astype(float)
    return df

def draw_figure_1():
    df = load_data("mpi-results-2.csv")
    df = df.iloc[5:]
    
    df_old = load_data("mpi-results-1.csv")
    df_old = df_old.iloc[5:]
    
    fig, ax = plt.subplots(figsize = (10, 6))
    
    cum_sched = df["t_IO"] + df["t_sched"]
    cum_calc_total = cum_sched + df["t_calc"]
    
    # Plot new optimized performance
    ax.fill_between(df["proc"], 0.01, cum_sched, alpha = 0.55, label = "Optimized\nScheduling\nand I/O", color = "C1", zorder = 0)
    ax.fill_between(df["proc"], cum_sched, cum_calc_total, alpha = 0.55, label = "Optimized\nCalculation", color = "C2", zorder = 0)
    ax.plot(df["proc"], cum_sched, marker = "o", zorder = 1, color = "C1")
    ax.plot(df["proc"], cum_calc_total, marker = "o", zorder = 1, color = "C2")
    
    # Plot old total runtime performance
    ax.plot(df_old["proc"], df_old["t_sched"]+df_old["t_IO"], linestyle = "--", label = "Unoptimized\nScheduling\nand I/O", zorder = 0, alpha = 1)
    ax.plot(df_old["proc"], df_old["t_total"], linestyle = "--", label = "Unoptimized\nCalculation", zorder = 0, color = "red", alpha = 1)
    
    ax.legend(title = "Runtime Split", ncols = 2, framealpha = 1,)
    
    ax.set_yscale("log", base = 10)
    ax.set_ylim(0.01, 10)
    ax.set_xlim(64, 160)
    x_ticks = list(df["proc"])
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -2)
    ax.set_axisbelow(True)
    ax.set_xticks(x_ticks)
    ax.set_xlabel("Processes")
    ax.set_ylabel("Runtime log(Seconds)")
    
    plt.savefig("figures/ex-3.png", dpi = 200)
    
def draw_figure_2():
    df = load_data("mpi-results-2.csv")
    
    ts = df["t_total"][0]
    
    df["sp"] = df["t_total"].apply(
        lambda tp: ts / tp 
    )
    
    df_old = load_data("mpi-results-1.csv")
    
    ts_old = df_old["t_total"][0]
    
    df_old["sp_old"] = df_old["t_total"].apply(
        lambda tp: ts_old / tp
    )
    
    fig, ax = plt.subplots(figsize = (10, 6))
    
    plt.plot(df["proc"], df["sp"], marker = "o", label = "Optimized", zorder = 2,)
    plt.plot(df_old["proc"], df_old["sp_old"], color = "orange", linewidth = 2, alpha = 1, label = "Unoptimized", linestyle = "-.", zorder = 1)
    plt.plot((0,160), (0,160), label = "Theoretical", color = "red", linestyle = "--", zorder = 0)
    
    ax.legend(title="Speedup", framealpha = 1)

    ax.set_xlim(21, 160)
    ax.set_ylim(20, 60)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)
    ax.set_axisbelow(True)
    
    ax.set_xlabel("Processes")
    ax.set_ylabel("Speedup")
    
    plt.savefig("figures/ex-4.png", dpi = 200)

if __name__ == "__main__":
    draw_figure_1()
    draw_figure_2()