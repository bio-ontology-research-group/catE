with open("params_ppi.txt", "w") as f:
    for weight_decay in [0, 0.0001, 0.001]:
        for batch_size in [2048, 4096, 8192]:
            for dim in [128, 256]:
                for margin in [0,0.02, 0.04, 0.06, 0.08]:
                    f.write(f"{dim} {margin} {weight_decay} {batch_size}\n")


print("Done")
