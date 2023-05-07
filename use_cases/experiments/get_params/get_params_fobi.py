with open("params_fobi.txt", "w") as f:
    for weight_decay in [0, 0.0001, 0.001]:
        for batch_size in [512, 1024, 2048]:
            for dim in [16, 32, 64]:
                for margin in [0, 0.04,0.08]:
                    f.write(f"{dim} {margin} {weight_decay} {batch_size}\n")


print("Done")
