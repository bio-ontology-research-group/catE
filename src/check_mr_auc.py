import sys

mr = float(sys.argv[1])
auc = float(sys.argv[2])

auc_diff = 1-auc
print(mr/auc_diff)
