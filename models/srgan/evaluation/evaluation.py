import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

metrics_dict = {"loss_D":'Discriminator Loss', "loss_G":'Generator Loss',
                'ssim_avg':'Structural Similarity Index Measure Average',
                'psnr_avg':'Peak signal-to-noise Ratio Average',
                'nrmse_avg':'Normalized Root Mean Squared Error Average'}

def load_csv(file_path):
    df = pd.read_csv(file_path,header=None,
                        names=['Epoch','Batch','loss_D','loss_G',
                               'ssim_1','ssim_2','ssim_3','ssim_4',
                               'psnr_1','psnr_2','psnr_3','psnr_4',
                               'nrmse_1','nrmse_2','nrmse_3','nrmse_4'])

    df['ssim_avg'] = df[['ssim_1','ssim_2', 'ssim_3' ,'ssim_4']].mean(axis=1)
    df['psnr_avg'] = df[['psnr_1','psnr_2', 'psnr_3' ,'psnr_4']].mean(axis=1)
    df['nrmse_avg'] = df[['nrmse_1','nrmse_2', 'nrmse_3' ,'nrmse_4']].mean(axis=1)

    metrics = ["loss_D", "loss_G",'ssim_avg','psnr_avg','nrmse_avg']
    red_df = df[metrics]
    return  red_df

def plot(ind_list:list, df:pd.DataFrame, metrics_dict=metrics_dict):
    color = ['blue','green','red','gray','orange']
    for i,key in enumerate(metrics_dict.keys()):
        plt.figure(figsize=(10,5))
        title = 'Plot of the ' + metrics_dict[key]
        plt.title(title)

        
        plt.plot(ind_list, df[key], color=color[i])
        plt.show()
        
def diff_plot(df_model1, df_model2, ind_list,metrics_dict=metrics_dict):
    color = ['blue','green','red','gray','orange']
    for i,key in enumerate(metrics_dict.keys()):
        diff = df_model1[key]-df_model2[key]
        
        plt.figure(figsize=(10,5))
        title = 'Plot of the ' + metrics_dict[key]
        plt.title(title)

        
        plt.plot(ind_list, diff, color=color[i])
        plt.show()

def stats(df):
    
    metrics = ["loss_D", "loss_G",'ssim_avg','psnr_avg','nrmse_avg']
    averages = {}
    
    for metric in metrics:
        averages[metric] = df[metric].mean()
        
    return averages
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    