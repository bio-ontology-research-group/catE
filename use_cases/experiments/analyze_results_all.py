import pandas as pd
import sys
import click as ck

def analyze_result_metric(file_path, metric, criterion='max'):
    header = ['embed_dim', 'margin', 'reg', 'batch_size', 'mr', 'mrr', 'h1', 'h3', 'h10', 'h100', 'auc', 'fmr', 'fmrr', 'fh1', 'fh3', 'fh10', 'fh100', 'fauc']
                    
    df = pd.read_csv(file_path, header=None, names=header)
        
    best = df.loc[df[metric].idxmax() if criterion == 'max' else df[metric].idxmin()].to_frame().T

    for col in header[4:]:
        if col in best.columns:
            if col in ['mr', 'mrr', 'fmr', 'fmrr']:
                best[col] = best[col].apply(lambda x: round(x, 2))
            else:
                best[col] = best[col].apply(lambda x: round(x*100, 2))
    return best


def get_graph_metrics(filename, final_metric):
        
    all_metrics = False

    if all_metrics:
        best_mr = analyze_result_metric(filename, "mr", criterion="min")
        best_mrr = analyze_result_metric(filename, "mrr", criterion="max")
        best_h1 = analyze_result_metric(filename, "h1")
        best_h3 = analyze_result_metric(filename, "h3")
        best_h10 = analyze_result_metric(filename, "h10")
        best_h100 = analyze_result_metric(filename, "h100")
        best_auc = analyze_result_metric(filename, "auc")
        best_fmr = analyze_result_metric(filename, "fmr", criterion="min")
        best_fmrr = analyze_result_metric(filename, "fmrr")
        best_fh1 = analyze_result_metric(filename, "fh1")
        best_fh3 = analyze_result_metric(filename, "fh3")
        best_fh10 = analyze_result_metric(filename, "fh10")
        best_fh100 = analyze_result_metric(filename, "fh100")
        best_fauc = analyze_result_metric(filename, "fauc")
        all_res = pd.concat([best_mr, best_mrr, best_h1, best_h3, best_h10, best_h100, best_auc, best_fmr, best_fmrr, best_fh1, best_fh3, best_fh10, best_fh100, best_fauc], axis=0)

        swap_list = ["embed_dim", "margin", "reg", "batch_size", "mrr", "mr", "h1", "h3", "h10", "h100", "auc", "fmrr", "fmr", "fh1", "fh3", "fh10", "fh100", "fauc"]
        
        all_res = all_res.reindex(columns=swap_list)
        print(all_res)
    else:
        best_h1 = analyze_result_metric(filename, final_metric)
        all_res = best_h1
        
        swap_list = ["embed_dim", "margin", "reg", "batch_size", "mrr", "fmrr", "mr", "h1", "h3", "h10", "h100", "auc", "fmr", "fh1", "fh3", "fh10", "fh100", "fauc"]
        
        all_res = all_res.reindex(columns=swap_list)
        print(all_res)
        all_res = list(all_res.iloc[0])
        tex_str = " & ".join([str(x) for x in all_res])
        print(tex_str)
        



if __name__ == "__main__":

    filename = sys.argv[1]
    final_metric = str(sys.argv[2])
      
    get_graph_metrics(filename, final_metric)
