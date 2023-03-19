import pandas as pd
import sys
import click as ck

def analyze_result_metric(file_path, metric, criterion="max", auc_threshold=None, h1_threshold=None):
    #header = ["embed_dim", "margin", "reg", "batch_size", "lr", "mr", "mrr", "h1", "h10", "h100","auc"]
    header = ["embed_dim", "margin", "reg", "batch_size", "lr", "mr", "mrr", "h1", "h10", "h100","auc", "fmr", "fmrr", "fh1", "fh10", "fh100", "fauc"]

    df = pd.read_csv(filename, header=None, names=header)
    df = df[df["auc"] > auc_threshold]
    #df = df[df['h1'] > h1_threshold]
                                                        
    if criterion == "max":
        best = df.loc[df[metric].idxmax()].to_frame().T
    elif criterion == "min":
        best = df.loc[df[metric].idxmin()].to_frame().T
        
    best["mr"] = best["mr"].apply(lambda x: round(x, 2))
    best["mrr"] = best["mrr"].apply(lambda x: round(x, 2))
    best["h1"] = best["h1"].apply(lambda x: round(x*100, 2))
    best["h10"] = best["h10"].apply(lambda x: round(x*100, 2))
    best["h100"] = best["h100"].apply(lambda x: round(x*100, 2))
    best["auc"] = best["auc"].apply(lambda x: round(x*100, 2))
    best["fmr"] = best["fmr"].apply(lambda x: round(x, 2))
    best["fmrr"] = best["fmrr"].apply(lambda x: round(x, 2))
    best["fh1"] = best["fh1"].apply(lambda x: round(x*100, 2))
    best["fh10"] = best["fh10"].apply(lambda x: round(x*100, 2))
    best["fh100"] = best["fh100"].apply(lambda x: round(x*100, 2))
    best["fauc"] = best["fauc"].apply(lambda x: round(x*100, 2))
    return best
    


def get_graph_metrics(filename, auc_threshold, h1_threshold):
        
    all_metrics = False

    if all_metrics:
        best_mr = analyze_result_metric(filename, "mr", criterion="min", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_mrr = analyze_result_metric(filename, "mrr", criterion="max", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_h1 = analyze_result_metric(filename, "h1", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_h10 = analyze_result_metric(filename, "h10", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_h100 = analyze_result_metric(filename, "h100", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_auc = analyze_result_metric(filename, "auc", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fmr = analyze_result_metric(filename, "fmr", criterion="min", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fmrr = analyze_result_metric(filename, "fmrr", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fh1 = analyze_result_metric(filename, "fh1", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fh10 = analyze_result_metric(filename, "fh10", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fh100 = analyze_result_metric(filename, "fh100", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        best_fauc = analyze_result_metric(filename, "fauc", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        all_res = pd.concat([best_mr, best_mrr, best_h1, best_h10, best_h100, best_auc, best_fmr, best_fmrr, best_fh1, best_fh10, best_fh100, best_fauc], axis=0)
        swap_list = ["embed_dim", "margin", "reg", "batch_size", "lr", "mrr", "mr", "h1", "h10", "h100", "auc", "fmrr", "fmr", "fh1", "fh10", "fh100", "fauc"]
        all_res = all_res.reindex(columns=swap_list)
        print(all_res)
    else:
        best_h1 = analyze_result_metric(filename, "auc", auc_threshold=auc_threshold, h1_threshold=h1_threshold)
        all_res = best_h1
        swap_list = ["embed_dim", "margin", "reg", "batch_size", "lr", "mrr", "mr", "h1", "h10", "h100", "auc", "fmrr", "fmr", "fh1", "fh10", "fh100", "fauc"]
        all_res = all_res.reindex(columns=swap_list)
        print(all_res)
        all_res = list(all_res.iloc[0])
        tex_str = " & ".join([str(x) for x in all_res])
        print(tex_str)
        



if __name__ == "__main__":

    filename = sys.argv[1]
    auc_threshold = float(sys.argv[2])
    h1_threshold = float(sys.argv[3])

    get_graph_metrics(filename, auc_threshold, h1_threshold)
