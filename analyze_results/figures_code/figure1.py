import numpy as np
base = '/home2/ebrahim/beyond-brainscore/'
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import sys
sys.path.append(base)
from plotting_functions import plot_across_subjects
import seaborn as sns
import pandas as pd
from trained_results_funcs import find_best_layer, find_best_sigma

noL2_arr = [False, True]
shuffled_arr = [False, True]
dataset_arr = ['pereira', 'fedorenko', 'blank']
perf_arr = ['pearson_r', 'out_of_sample_r2']
feature_extraction = ['', '-sp', '-mp']

save_best_sigma = {}
save_best_layer = {}

for perf in perf_arr:
    for noL2 in noL2_arr:
        for shuffled in shuffled_arr:
            
            fig, ax = plt.subplots(1, 3, figsize=(15, 6))
            
            for dataset in dataset_arr:
                
                if noL2:
                    noL2_str = '_noL2'
                    noL2_save_str = '_lin'
                else:
                    noL2_str = ''
                    noL2_save_str = '_ridge'
                    
                if shuffled:
                    shuffled_str = '_shuffled'
                    shuffled_save_str = '_shuffled'
                    
                else:
                    shuffled_str = ''
                    shuffled_save_str = '_contig'
                    

                
                resultsPath_dataset_nonshuffled = f'/data/LLMs/brainscore/results_{dataset}'
                if shuffled:
                    resultsPath_dataset = f'/data/LLMs/brainscore/results_{dataset}/shuffled'
                else:
                    resultsPath_dataset = resultsPath_dataset_nonshuffled
                    
                data_processed_folder = f'/data/LLMs/data_processed/{dataset}/dataset'
                figurePath = '/home2/ebrahim/beyond-brainscore/analyze_results/figures_code/figures/new_figures/figure1/'
                figurePath = f'{figurePath}{perf}/'

        
                # load information regarding number of voxels, subjects, and functional network localization for each experiment into a dictionary
                if dataset ==  'pereira':

                    exp = ['243', '384']

                    br_labels_dict = {}
                    num_vox_dict = {}
                    subjects_dict = {}
                    for e in exp:

                        bre = np.load(f'{data_processed_folder}/networks_{e}.npy', allow_pickle=True)
                        br_labels_dict[e] = bre
                        num_vox_dict[e] = bre.shape[0]
                        subjects_dict[e] = np.load(f"{data_processed_folder}/subjects_{e}.npy", allow_pickle=True)
                        
                    lang_indices_384 = np.argwhere(br_labels_dict['384'] == 'language').squeeze()
                    lang_indices_243 = np.argwhere(br_labels_dict['243'] == 'language').squeeze()
                    
                else:
                    subjects_arr  = np.load(f"{data_processed_folder}/subjects.npy", allow_pickle=True)
                    

                if shuffled:
                    # script crashes at sigma of 4.3 in fed due to an issue with linear reg converging
                    if dataset == 'fedorenko' and shuffled:
                        sigma_values = np.linspace(0.1, 4.2, 42)
                    else:
                        sigma_values = np.linspace(0.1, 4.8, 48)
                                
                    if dataset == 'pereira':
                        sigma_perf_dict_384, best_sigma_384, OASM_perf_best_sigma_384 = find_best_sigma(sigma_values, noL2_str=noL2_str, exp='_384', subjects=subjects_dict['384'], 
                                                                        resultsPath=resultsPath_dataset, dataset=dataset, selected_network_indices=lang_indices_384, 
                                                                        perf=perf)   
                        sigma_perf_dict_243, best_sigma_243, OASM_perf_best_sigma_243 = find_best_sigma(sigma_values, noL2_str=noL2_str, exp='_243', subjects=subjects_dict['243'], 
                                                                        resultsPath=resultsPath_dataset, dataset=dataset, selected_network_indices=lang_indices_243, 
                                                                        perf=perf)
                        

                    
                        save_best_sigma[f"{dataset}_384_{perf}{shuffled_save_str}{noL2_save_str}"] = best_sigma_384
                        save_best_sigma[f"{dataset}_243_{perf}{shuffled_save_str}{noL2_save_str}"] = best_sigma_243
                    else:
                        sigma_perf_dict, best_sigma, OASM_perf_best_sigma = find_best_sigma(sigma_values, noL2_str=noL2_str, exp='', 
                                                            subjects=subjects_arr, resultsPath=resultsPath_dataset, dataset=dataset, perf=perf)
                        
                        save_best_sigma[f"{dataset}_{perf}{shuffled_save_str}{noL2_save_str}"] = best_sigma
                        
                    
                    #if dataset == 'pereira':
                    #    plt.plot(sigma_perf_dict_384.keys(), sigma_perf_dict_384.values(), label='384')
                    #    plt.plot(sigma_perf_dict_243.keys(), sigma_perf_dict_243.values(), label='243')
                    #else:
                    #    plt.plot(sigma_perf_dict.keys(), sigma_perf_dict.values())
                        
                    #plt.legend()
                    #plt.xlabel("Sigma values")
                    #plt.ylabel("Median pearson r across language voxels")
                    #plt.savefig(f"{figurePath}across_layer/across_layer_OASM_{dataset}{noL2_str}{shuffled_str}")
                    #plt.close()
                
                else:
                    
                    if dataset == 'pereira':
                        
                        simple_perf_384 =  np.load(f'{resultsPath_dataset}/{dataset}_positional_WN_layer1_1{noL2_str}_384.npz')[perf]
                        simple_perf_243 =  np.load(f'{resultsPath_dataset}/{dataset}_positional_WN_layer1_1{noL2_str}_243.npz')[perf]
                        
                    if dataset == 'fedorenko':
        
                        simple_perf =  np.load(f'{resultsPath_dataset}/{dataset}_soft+grow_layer1_1{noL2_str}.npz')[perf]
                        
                
                results_dict_gpt2 = {'perf':[], 'subjects': [], 'Network': [], 
                                'Model': []}
                
                if dataset == 'pereira':
           
                    for fe in feature_extraction:
                        
                        gpt2_xl_384_dict, gpt2_xl_384_bl, gpt2_xl_384_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_384', 
                                                                        resultsPath=resultsPath_dataset, selected_network_indices=lang_indices_384, dataset=dataset, 
                                                                        subjects=subjects_dict['384'], perf=perf, feature_extraction=fe)
                        gpt2_xl_243_dict, gpt2_xl_243_bl, gpt2_xl_243_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='_243', 
                                                                        resultsPath=resultsPath_dataset, selected_network_indices=lang_indices_243, dataset=dataset, 
                                                                        subjects=subjects_dict['243'], perf=perf, feature_extraction=fe)
                        
                        results_dict_gpt2['perf'].extend(gpt2_xl_384_bl_perf)
                        results_dict_gpt2['perf'].extend(gpt2_xl_243_bl_perf)
                        results_dict_gpt2['subjects'].extend(subjects_dict['384'])
                        results_dict_gpt2['subjects'].extend(subjects_dict['243'])
                        results_dict_gpt2['Network'].extend(br_labels_dict['384'])
                        results_dict_gpt2['Network'].extend(br_labels_dict['243'])
                        results_dict_gpt2['Model'].extend(np.repeat(f'GPT2-XL{fe}', num_vox_dict['384']))
                        results_dict_gpt2['Model'].extend(np.repeat(f'GPT2-XL{fe}', num_vox_dict['243']))
                        
                        
                        
                        save_best_layer[f"{dataset}_384_{perf}{shuffled_save_str}{noL2_save_str}{fe}"] = gpt2_xl_384_bl
                        save_best_layer[f"{dataset}_243_{perf}{shuffled_save_str}{noL2_save_str}{fe}"] = gpt2_xl_243_bl
                        
            
                else:
                    for fe in feature_extraction:
                        gpt2_xl_dict, gpt2_xl_bl, gpt2_xl_bl_perf = find_best_layer(np.arange(0,49), noL2_str=noL2_str, exp='', 
                                                subjects=subjects_arr, resultsPath=resultsPath_dataset, dataset=dataset, perf=perf, feature_extraction=fe)
                        
                        num_brain_units = gpt2_xl_bl_perf.shape[0]
                        
                        results_dict_gpt2['perf'].extend(gpt2_xl_bl_perf)
                        results_dict_gpt2['subjects'].extend(subjects_arr)
                        results_dict_gpt2['Network'].extend(np.repeat('language', num_brain_units))
                        results_dict_gpt2['Model'].extend(np.repeat(f'GPT2-XL{fe}', num_brain_units))
                        
                        save_best_layer[f"{dataset}_{perf}{shuffled_save_str}{noL2_save_str}{fe}"] = gpt2_xl_bl
                        
                        
                results_dict_gpt2 = pd.DataFrame(results_dict_gpt2)
                results_dict_simple = None    

                if dataset == 'pereira':
                    
                    if shuffled:
                        results_dict_simple_384 = pd.DataFrame({'perf': OASM_perf_best_sigma_384, 'subjects': subjects_dict['384'], 
                                        'Network': br_labels_dict['384'], 'Model': np.repeat('OASM', num_vox_dict['384'])})
                        results_dict_simple_243 = pd.DataFrame({'perf': OASM_perf_best_sigma_243, 'subjects': subjects_dict['243'], 
                                        'Network': br_labels_dict['243'], 'Model': np.repeat('OASM', num_vox_dict['243'])})
                        
                    else:
                        results_dict_simple_384 = pd.DataFrame({'perf': simple_perf_384, 'subjects': subjects_dict['384'], 
                                        'Network': br_labels_dict['384'], 'Model': np.repeat('SP+SL', num_vox_dict['384'])})
                        results_dict_simple_243 = pd.DataFrame({'perf': simple_perf_243, 'subjects': subjects_dict['243'], 
                                        'Network': br_labels_dict['243'], 'Model': np.repeat('SP+SL', num_vox_dict['243'])})
                        
                    results_dict_simple = pd.concat((results_dict_simple_384, results_dict_simple_243))
    
                    
                else:
                    num_brain_units = gpt2_xl_bl_perf.shape[0]
                    
                    if shuffled:
                        results_dict_simple = pd.DataFrame({'perf': OASM_perf_best_sigma, 'subjects': subjects_arr, 'Network': np.repeat('language', num_brain_units),
                                                    'Model': np.repeat('OASM', num_brain_units)})
                    else:
                        
                        if dataset == 'fedorenko':
                            results_dict_simple = pd.DataFrame({'perf': simple_perf, 'subjects': subjects_arr, 'Network': np.repeat('language', num_brain_units),
                                                    'Model': np.repeat('WP', num_brain_units)})


                if results_dict_simple is not None:
                    results_simple_gpt2xl = pd.concat((results_dict_simple, results_dict_gpt2))
                
                if perf == 'pearson_r':
                    median = True
                    clip_zero = False
                    perf_str = 'Pearson r'
                    plot_xlabel = False
                else:
                    median = False
                    clip_zero = True
                    perf_str = r'$R^2$'
                    plot_xlabel = True

                if shuffled:
                    cidx_p = 9
                    cidx_fb = 9
                else:
                    cidx_p = 7
                    cidx_fb = 6
                    
                
                if perf == 'pearson_r' and shuffled:
                    ymax = 0.7
                if perf == 'pearson_r' and shuffled == False:
                    ymax = 0.4
                if perf == 'out_of_sample_r2' and shuffled:
                    ymax = 0.3
                if perf == 'out_of_sample_r2' and shuffled == False:
                    ymax = 0.12
                    
                  
                # Define shades of blue and an orange color
                if shuffled:
                    palette = sns.color_palette(["#1E90FF", "#4169E1", "#0000CD", "#FFA500"]) 
                    
                elif dataset == 'pereira':
                    palette = sns.color_palette(["#1E90FF", "#4169E1", "#0000CD", "#008000"])  # "#008000" is green
                    
                elif dataset == 'fedorenko':
                    palette = sns.color_palette(["#1E90FF", "#4169E1", "#0000CD", "#FF69B4"]) 
                      
                if dataset == 'pereira':
                    index = 0
                    remove_y_axis = False
                    dataset_label = 'Pereira2016'
        
                if dataset == 'fedorenko':
                    index = 1
                    remove_y_axis = True 
                    dataset_label = 'Fed2016'
            
                if dataset == 'blank':
                    index = 2
                    remove_y_axis = True
                    dataset_label = 'Blank2014'
                    
    
                plot_legend = False

                if dataset == 'pereira':

                    
                    subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_simple_gpt2xl.copy(), figurePath=figurePath,  selected_networks=['language'],
                                                            dataset=dataset_label, saveName=f'{dataset}{noL2_str}{shuffled_str}_both', 
                                                            yticks=[0, ymax], order=['language'], clip_zero=clip_zero, color_palette=palette, 
                                                            draw_lines=False, ms=15, plot_legend=plot_legend, 
                                                            plot_legend_under=False, width=0.7, median=median, ylabel_str=perf_str, legend_fontsize=30, ax_select=ax[index],
                                                            remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel)
                else:
                    
                    if shuffled or dataset == 'fedorenko':
                    
                          
                        #max_val = round(results_simple_gpt2xl['perf'].max() + 0.1*results_simple_gpt2xl['perf'].max(), 2)
                        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_simple_gpt2xl.copy(), figurePath=figurePath, selected_networks=['language'],
                                                                dataset=dataset_label, saveName=f'{dataset}{noL2_str}{shuffled_str}', 
                                                                yticks=[0, ymax], order=['language'], clip_zero=clip_zero, color_palette=palette, 
                                                                draw_lines=False, ms=15, plot_legend=plot_legend, 
                                                                plot_legend_under=False, width=0.7, median=median, ylabel_str='', legend_fontsize=30, ax_select=ax[index],
                                                                remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel)
                    else:
                        
                        subject_avg_pd, dict_pd_merged, dict_pd_with_all = plot_across_subjects(results_dict_gpt2.copy(), figurePath=figurePath, selected_networks=['language'],
                                                                dataset=dataset_label, saveName=f'{dataset}{noL2_str}{shuffled_str}', 
                                                                yticks=[0, ymax], order=['language'], clip_zero=clip_zero, color_palette=palette, 
                                                                draw_lines=False, ms=15, plot_legend=False, 
                                                                plot_legend_under=False, width=0.7, median=median, ylabel_str='', ax_select=ax[index], remove_yaxis=remove_y_axis, plot_xlabel=plot_xlabel)
                
                
                # for some reason that I don't understand the y axis is not removed through the function with the Fed plots
                # so I just do it manually here (fed is always index 1)
                ax[1].spines['left'].set_visible(False)   # Hide the left spine
                ax[1].yaxis.set_visible(False)            # Hide the y-axis
                ax[1].set_yticks([])                      # Remove yticks
                ax[1].set_yticklabels([])                 # Remove ytick labels (if any)
                
                if noL2:
                    ax[1].legend(loc="upper left")  # Adjust 'loc' as needed
                    
                plt.legend(fontsize=20, frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
                
                
                # also realigning the x axis here, just easier to do it outside the function 
                for i in range(3):
                    ax[i].set_ylim(0, ymax)                  # Set y-limits to be the same
                    ax[i].spines['bottom'].set_position(('data', 0))  # Place the x-axis at y=0
                    ax[i].set_ylabel('')
                    ax[i].set_xlabel('')

                fig.savefig(f'{figurePath}figure1{noL2_str}{shuffled_str}.png')
                fig.savefig(f'{figurePath}figure1{noL2_str}{shuffled_str}.pdf',  bbox_inches='tight')
                                            

                            


np.savez('best_layer_sigma_info/best_sigma', **save_best_sigma)
np.savez('best_layer_sigma_info/best_gpt2xl_layer', **save_best_layer)










