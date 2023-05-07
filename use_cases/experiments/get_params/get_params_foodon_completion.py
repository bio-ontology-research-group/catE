with open("params_foodon_completion.txt", "w") as f:
    for weight_decay in [0, 0.0001, 0.001]:
        for batch_size in [4096, 8192, 16834]:
            for dim in [32, 64, 128]:
                for margin in [0,0.02, 0.04]:
                    f.write(f"{dim} {margin} {weight_decay} {batch_size}\n")


print("Done")
