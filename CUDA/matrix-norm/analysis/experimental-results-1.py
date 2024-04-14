# libraries
import pandas as pd
import matplotlib.pyplot as plt

def load_data():
    df = pd.read_csv("data/cuda-1.csv", header = 0)
    for col in df.columns:
        if col != "N":
            df[col] = df[col].astype(float)
    return df

def draw_figure():
    df = load_data()
    
    fig, ax = plt.subplots(figsize = (10, 6))
    
    ax.plot(df["N"], df["Serial"], label = "Serial", marker = "o")
    ax.plot(df["N"], df["CUDA"], label = "CUDA", marker = "o")
    
    ax.legend(framealpha = 1,)
    
    ax.set_yscale("log", base = 10)
    ax.set_xlim(0, 45000)
    ax.set_ylim(1, 100000)
    ax.grid(which='both', linestyle='-', linewidth=0.5, zorder = -1)
    ax.set_xlabel("N")
    ax.set_ylabel("Runtime log(milliseconds)")
    ax.set_axisbelow(True)
    
    plt.savefig("figures/cuda-1.png", dpi = 200)
    
if __name__ == "__main__":
    draw_figure()