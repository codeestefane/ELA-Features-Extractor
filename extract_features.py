import os
import pandas as pd
from pflacco.classical_ela_features import *
from tqdm import tqdm

amostras = [10, 20, 30, 40, 50, 100, 250, 500, 750, 1000, 2500, 5000, 10000, 15000, 20000, 30000]
seeds = [3, 5, 7, 13, 27, 35, 42, 66, 72, 111]

def extractor_features(X, y, dados):
    ela_meta_df = pd.DataFrame({'ela_meta.lin_simple.adj_r2': [], 'ela_meta.lin_simple.intercept': [], 'ela_meta.lin_simple.coef.min': [], 'ela_meta.lin_simple.coef.max': [], 'ela_meta.lin_simple.coef.max_by_min': [], 'ela_meta.lin_w_interact.adj_r2': [], 'ela_meta.quad_simple.adj_r2': [], 'ela_meta.quad_simple.cond': [], 'ela_meta.quad_w_interact.adj_r2': [], 'ela_meta.costs_runtime': []})
    ela_distr_df = pd.DataFrame({'ela_distr.skewness': [], 'ela_distr.kurtosis': [], 'ela_distr.number_of_peaks': [], 'ela_distr.costs_runtime': []})
    ela_cm_angle_df = pd.DataFrame({'cm_angle.dist_ctr2best_mean': [], 'cm_angle.dist_ctr2best_sd': [], 'cm_angle.dist_ctr2worst_mean': [], 'cm_angle.dist_ctr2worst_sd': [], 'cm_angle.angle_mean': [], 'cm_angle.angle_sd': [], 'cm_angle.y_ratio_best2worst_mean': [], 'cm_angle.y_ratio_best2worst_sd': [], 'cm_angle.costs_runtime': []})
    ela_cm_conv_df = pd.DataFrame({'cm_conv.convex.hard': [], 'cm_conv.concave.hard': [], 'cm_conv.convex.soft': [], 'cm_conv.concave.soft': [], 'cm_conv.costs_runtime': []})
    ela_cm_grad_df = pd.DataFrame({'cm_grad.mean': [], 'cm_grad.sd': [], 'cm_grad.costs_runtime': []})
    ela_ic_df = pd.DataFrame({'ic.h_max': [], 'ic.eps_s': [], 'ic.eps_max': [], 'ic.eps_ratio': [], 'ic.m0': [], 'ic.costs_runtime': []})
    ela_nbc_df = pd.DataFrame({'nbc.nn_nb.sd_ratio': [], 'nbc.nn_nb.mean_ratio': [], 'nbc.nn_nb.cor': [], 'nbc.dist_ratio.coeff_var': [], 'nbc.nb_fitness.cor': [], 'nbc.costs_runtime': []})
    ela_limo_df = pd.DataFrame({'limo.avg_length': [], 'limo.avg_length_norm': [], 'limo.length_mean': [], 'limo.length_sd': [], 'limo.cor': [], 'limo.cor_norm': [], 'limo.ratio_mean': [], 'limo.ratio_sd': [], 'limo.sd_ratio_reg': [], 'limo.sd_ratio_norm': [], 'limo.sd_mean_reg': [], 'limo.sd_mean_norm': [], 'limo.costs_runtime': []})
    ela_pca_df = pd.DataFrame({'pca.expl_var.cov_x': [], 'pca.expl_var.cor_x': [], 'pca.expl_var.cov_init': [], 'pca.expl_var.cor_init': [], 'pca.expl_var_PC1.cov_x': [], 'pca.expl_var_PC1.cor_x': [], 'pca.expl_var_PC1.cov_init': [], 'pca.expl_var_PC1.cor_init': [], 'pca.costs_runtime': []})
    ela_disp_df = pd.DataFrame({'disp.ratio_mean_02': [], 'disp.ratio_mean_05': [], 'disp.ratio_mean_10': [], 'disp.ratio_mean_25': [], 'disp.ratio_median_02': [], 'disp.ratio_median_05': [], 'disp.ratio_median_10': [], 'disp.ratio_median_25': [], 'disp.diff_mean_02': [], 'disp.diff_mean_05': [], 'disp.diff_mean_10': [], 'disp.diff_mean_25': [], 'disp.diff_median_02': [], 'disp.diff_median_05': [], 'disp.diff_median_10': [], 'disp.diff_median_25': [], 'disp.costs_runtime': []})

    lowerBound = [min(dados['cost'].values), min(dados['gamma'].values)]
    upperBound = [max(dados['cost'].values), max(dados['gamma'].values)]

    try: 
        ela_meta = calculate_ela_meta(X, y)
        ela_meta_df = pd.concat([ela_meta_df, pd.DataFrame(ela_meta, index = [0])], ignore_index=True)
    except:
        ela_meta_df.loc[len(ela_meta_df)] = [None for i in range(len(ela_meta_df.columns))]

    try: 
        ela_distr = calculate_ela_distribution(X, y)
        ela_distr_df = pd.concat([ela_distr_df, pd.DataFrame(ela_distr, index = [0])], ignore_index=True)
    except:
        ela_distr_df.loc[len(ela_distr_df)] = [None for i in range(len(ela_distr_df.columns))]

    try: 
        ela_cm_angle = calculate_cm_angle(X, y, lowerBound, upperBound)
        ela_cm_angle_df = pd.concat([ela_cm_angle_df, pd.DataFrame(ela_cm_angle, index = [0])], ignore_index=True)
    except:
        ela_cm_angle_df.loc[len(ela_cm_angle_df)] = [None for i in range(len(ela_cm_angle_df.columns))]

    try: 
        ela_cm_conv = calculate_cm_conv(X, y, lowerBound, upperBound)
        ela_cm_conv_df = pd.concat([ela_cm_conv_df, pd.DataFrame(ela_cm_conv, index = [0])], ignore_index=True)
    except:
        ela_cm_conv_df.loc[len(ela_cm_conv_df)] = [None for i in range(len(ela_cm_conv_df.columns))]

    try: 
        ela_cm_grad = calculate_cm_grad(X, y, lowerBound, upperBound)
        ela_cm_grad_df = pd.concat([ela_cm_grad_df, pd.DataFrame(ela_cm_grad, index = [0])], ignore_index=True)
    except:
        ela_cm_grad_df.loc[len(ela_cm_grad_df)] = [None for i in range(len(ela_cm_grad_df.columns))]
        
    try: 
        ela_ic = calculate_information_content(X, y)
        ela_ic_df = pd.concat([ela_ic_df, pd.DataFrame(ela_ic, index = [0])], ignore_index=True)
    except:
        ela_ic_df.loc[len(ela_ic_df)] = [None for i in range(len(ela_ic_df.columns))]

    try: 
        ela_nbc = calculate_nbc(X, y)
        ela_nbc_df = pd.concat([ela_nbc_df, pd.DataFrame(ela_nbc, index = [0])], ignore_index=True)
    except:
        ela_nbc_df.loc[len(ela_nbc_df)] = [None for i in range(len(ela_nbc_df.columns))]

    try: 
        ela_limo = calculate_limo(X, y, lowerBound, upperBound)
        ela_limo_df = pd.concat([ela_limo_df, pd.DataFrame(ela_limo, index = [0])], ignore_index=True)
    except:
       ela_limo_df.loc[len(ela_limo_df)] = [None for i in range(len(ela_limo_df.columns))]

    try: 
        ela_pca = calculate_pca(X, y)
        ela_pca_df = pd.concat([ela_pca_df, pd.DataFrame(ela_pca, index = [0])], ignore_index=True)
    except:
       ela_pca_df.loc[len(ela_pca_df)] = [None for i in range(len(ela_pca_df.columns))]

    try: 
        ela_disp = calculate_dispersion(X, y)
        ela_disp_df = pd.concat([ela_disp_df, pd.DataFrame(ela_disp, index = [0])], ignore_index=True)
    except:
       ela_disp_df.loc[len(ela_disp_df)] = [None for i in range(len(ela_disp_df.columns))]

    return pd.concat([ela_meta_df, ela_distr_df, ela_cm_angle_df, ela_cm_conv_df, ela_cm_grad_df, ela_ic_df, ela_nbc_df, ela_limo_df, ela_pca_df, ela_disp_df], axis = 1)

files_path = os.listdir("classif.svm")

for x, i in enumerate(files_path):
    features_pflacco = []
    arquivo_atual = i.replace(".csv", "")

    for qtd_amostras in amostras:
        for seed in tqdm(seeds, desc = f"{x} - {arquivo_atual} | Qtd. amostras = {qtd_amostras}"):
            dados = pd.read_csv("classif.svm/" + i)
            sample_size = min(qtd_amostras, len(dados))
            dados_sample = dados.sample(n=sample_size, random_state=seed)

            X = dados_sample[['cost', 'gamma']]
            y = dados_sample['ber.test.mean']

            features_pflacco.append(pd.concat([pd.DataFrame({"file": arquivo_atual}, index = [0]), pd.DataFrame({"amostras": qtd_amostras}, index = [0]), pd.DataFrame({"seed": seed}, index = [0]), extractor_features(X, y, dados_sample)], axis = 1))

            if qtd_amostras == 30000:
                break
            
    features_pflacco = pd.concat([df for df in features_pflacco], axis = 0, ignore_index = True)
    os.makedirs("metafeatures", exist_ok=True)
    features_pflacco.to_csv("metafeatures/" + arquivo_atual + ".csv", index = False, encoding = 'utf-8')

