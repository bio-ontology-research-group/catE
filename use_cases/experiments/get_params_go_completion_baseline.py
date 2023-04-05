with open("params_go_completion_baseline.txt", "w") as f:
    for weight_decay in [0, 0.0001, 0.001]:
        for batch_size in [1024, 2048, 4096]:
            for dim in [32, 64, 128]:
                for margin in [0,0.02, 0.04]:
                    f.write(f"{dim} {margin} {weight_decay} {batch_size}\n")


print("Done")
