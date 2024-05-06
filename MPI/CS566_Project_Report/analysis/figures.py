# libraries
import pandas as pd
import matplotlib.pyplot as plt

width = 8
height = 4

def load_data1():
    df = pd.read_csv("model1_results.csv", header = 0)
    for col in df.columns:
        if col != "proc":
            df[col] = df[col].astype(float)
    return df

def load_data2():
    df = pd.read_csv("model2_results.csv", header = 0)
    for col in df.columns:
        if col != "proc":
            df[col] = df[col].astype(float)
    return df

def figure1():
    df1 = load_data1()
    df2 = load_data2()
    
    ts = df1["ttotal"][0]
    
    df1["sp"] = df1["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    df2["sp"] = df2["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    fig, ax = plt.subplots(figsize = (width, height))
    
    ax.set_title("Model 1 Run Time Summary")
    ax.plot(df1["proc"], df1["ttotal"], marker = "o", label = "Total", zorder = 5, linewidth = 1.00)
    ax.plot(df1["proc"], df1["tcomm"], marker = "x", label = "Communication", zorder = 4, linewidth = 1.00)
    ax.plot(df1["proc"], df1["tcalc"], marker = "+", label = "Calculation", zorder = 3, linewidth = 1.00)
    
 
    
    ax.hlines(y = ts, color = "g", xmin = 1, xmax = 16,linestyle = "--", label = "Serial", alpha = 0.5)
    ax.plot(df2["proc"], df2["ttotal"], linestyle = "--", color = "blue", label = "Model 2", alpha = 0.5)
    ax.hlines(y = 0.013115, color = "r", xmin = 1, xmax = 16,linestyle = "--", label = "Model 3", alpha = 0.5)
    
    x_ticks = list(df1["proc"])
    ax.set_ylim(1e-3, 1)
    ax.set_xticks(x_ticks)
    ax.set_yscale("log", base = 10)
    ax.set_xlim(0.5, 16.5)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)

    # ax.set_ylim(0, 0.06)
    ax.minorticks_off()
    ax.legend(framealpha = 1.0, ncols = 2)
    # plt.show()
    ax.set_xlabel("Processes            ")
    ax.set_ylabel("Runtime log(Seconds)")
    plt.tight_layout()
    plt.savefig("figures/fig1.png", dpi = 200)
    
def figure2():
    df1 = load_data1()
    df2 = load_data2()
    
    ts = df1["ttotal"][0]
    
    df1["sp"] = df1["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    df2["sp"] = df2["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    fig, ax = plt.subplots(figsize = (width, height))
    
    ax.set_title("Model 2 Run Time Summary")
    ax.plot(df2["proc"], df2["ttotal"], marker = "o", label = "Total", zorder = 5, linewidth = 1.00)
    ax.plot(df2["proc"], df2["tcomm"], marker = "x", label = "Communication", zorder = 4, linewidth = 1.00)
    ax.plot(df2["proc"], df2["tcalc"], marker = "+", label = "Calculation", zorder = 3, linewidth = 1.00)
    
    ax.hlines(y = ts, color = "g", xmin = 1, xmax = 16,linestyle = "--", label = "Serial", alpha = 0.5)
    ax.plot(df1["proc"], df1["ttotal"], linestyle = "--", color = "blue", label = "Model 1", alpha = 0.5)
    ax.hlines(y = 0.013115, color = "r", xmin = 1, xmax = 16, linestyle = "--", label = "Model 3", alpha = 0.5)
    
    x_ticks = list(df1["proc"])
    ax.set_xticks(x_ticks)
    ax.set_yscale("log", base = 10)
    ax.set_xlim(0.5, 16.5)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)

    ax.set_ylim(10e-4, 1)
    ax.minorticks_off()
    ax.legend(framealpha = 1.0, ncols = 2)
    ax.set_xlabel("Processes            ")
    ax.set_ylabel("Runtime log(Seconds)")
    # plt.show()
    plt.tight_layout()
    plt.savefig("figures/fig2.png", dpi = 200)
    
def figure3():
    df1 = load_data1()
    df2 = load_data2()
    
    ts = df1["ttotal"][0]
    
    df1["sp"] = df1["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    df2["sp"] = df2["ttotal"].apply(
        lambda tp: ts / tp
    )
    
    fig, ax = plt.subplots(figsize = (width, height))
    
    ax.set_title("Speedup Comparison")
    ax.plot(df1["proc"], df1["sp"], marker = "o", label = "Model 1", zorder = 6, linewidth = 1.00)
    ax.plot(df2["proc"], df2["sp"], marker = "o", label = "Model 2", zorder = 5, linewidth = 1.00)
    ax.plot(8, ts / 0.013115, marker = "o", color = "red", linestyle = "--", label = "Model 3")
    ax.plot([1,8],[1,8], label = "Theoretical", linestyle = "--", alpha = 0.5)
    ax.hlines(y = ts / 0.013115, color = "r", xmin = 1, xmax = 16, linestyle = "--", alpha = 0.5)
    
    x_ticks = list(df1["proc"])
    ax.set_xticks(x_ticks)
    # ax.set_yscale("log", base = 10)
    ax.set_xlim(0.5, 16.5)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)

    ax.set_ylim(0, 4)
    ax.minorticks_off()
    ax.legend(framealpha = 1.0, ncols = 2)
    ax.set_xlabel("Processes            ")
    ax.set_ylabel("Speedup")
    # plt.show()
    plt.tight_layout()
    plt.savefig("figures/fig3.png", dpi = 200)

if __name__ == "__main__":
    
    figure1()
    figure2()
    figure3()
    # draw_figure_1()
    # draw_figure_2()
    # draw_figure_3()