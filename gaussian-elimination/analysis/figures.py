import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def speedup(N: int, procs: int) -> float:
    sp = (2*procs*N**2+3*procs*N)/(N**2+2*N+procs**2*N+procs**2)
    return sp

def efficiency(N: int, procs: int) -> float:
    ef = speedup(N, procs)/procs
    return ef/2

def serial_time(N: int) -> float:
    return 2*N**3+3*N**2

def parallel_time(N: int, procs: int) -> float:
    tp = (N**3+2*N**3+procs**2*N**2+procs**2*N)/procs
    return tp

def data_points() -> dict:
    data = {
        "N": np.arange(0,2050,50),
        "procs":[2, 4, 8, 16, 32],
    }
    return data

def generate_data() -> dict:
    data = data_points()
    
    data["ts"] = [serial_time(n) for n in data["N"]]
    for proc in data["procs"]:
        tp = [parallel_time(n, proc) for n in data["N"]]
        data[f"tp_proc_{proc}"] = tp
        
    for n in data["N"]:
        sps = [speedup(n, proc) for proc in data["procs"]]
        data[f"sp_n_{n}"] = sps
        efs = [efficiency(n, proc) for proc in data["procs"]]
        data[f"ef_n_{n}"] = efs
    
    return data

def draw_figure1() -> None:
    data = generate_data()
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    
    for key in data.keys():
        if key.startswith("sp_"):
            ax1.plot(
                data["procs"],
                data[key],
                label=f"Procs = {key.split("_")[-1]}",
                marker = "o"
            )
        if key.startswith("ef_"):
            ax2.plot(
                data["procs"],
                data[key],
                label=f"Procs = {key.split("_")[-1]}"
            )
    
    legend1 = ax1.legend(title="Speedup")
    ax1.add_artist(legend1)
    legend2 = ax2.legend(title="Efficiency")
    ax2.add_artist(legend2)
    
    ax1.set_ylabel("speedup")
    ax2.set_ylabel("efficiency")
    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    plt.show()

def draw_figure2():
    data = generate_data()
    
    fig, ax1 = plt.subplots()
    
    ax1.plot(
        data["N"],
        data["ts"],
        label=f"Serial",
    )
    
    for key in data.keys():
        if key.startswith("tp_"):
            ax1.plot(
                data["N"],
                data[key],
                label=f"Procs = {key.split("_")[-1]}",
            )
    
    legend1 = ax1.legend(title="Processors")
    ax1.add_artist(legend1)
    
    ax1.set_ylabel("Runtime")
    ax1.grid(which='both', linestyle='-', linewidth=0.5)
    plt.show()

if __name__ == "__main__":
    draw_figure2()