import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def serial_time(N: int) -> float:
    return 2*N**3+3*N**2

def parallel_time(N: int, procs: int) -> float:
    tp = (2*N**3+3*N**2)/procs
    return tp

def data_points() -> dict:
    data = {
        "N": [100, 1000, 2000, 3000, 4000, 5000],
        "procs":[2, 4, 8, 16, 32, 64, 128],
    }
    return data

def generate_data() -> dict:
    data = data_points()
    for n in data["N"]:
        tp = [parallel_time(n, proc) for proc in data["procs"]]
        data[f"tp_N={n}"] = tp
    data["str_procs"] = [str(proc) for proc in data["procs"]]
    return data

def draw_figure() -> None:
    data = generate_data()
    
    fig, ax1 = plt.subplots()
    
    for key in data.keys():
        if key.startswith("tp_"):
            ax1.plot(
                data["str_procs"],
                data[key],
                label=f"N = {key.split("=")[-1]}",
                marker = "^"
            )
    
    ax1.legend()
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title="Workload")
    ax1.set_yscale("log", base = 10)
    ax1.set_ylabel("log(Operations)")
    ax1.set_xlabel("Processors")
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    plt.savefig("./figures/parallel-runtime.png", dpi = 200, bbox_inches="tight")
    plt.show()
    
if __name__ == "__main__":
    draw_figure()