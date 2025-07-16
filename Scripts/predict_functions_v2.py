#! /usr/bin/python3

#to predict
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import matthews_corrcoef, f1_score, confusion_matrix, accuracy_score, roc_auc_score, jaccard_score, zero_one_loss, hamming_loss
from keras.models import load_model
from pickle import dump, load
import keras.backend as K

def matthews_correlation_coefficient(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1 - y_pred), 0, 1)))

    num = tp * tn - fp * fn
    den = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return num / K.sqrt(den + K.epsilon())

def get_file(path):
    data_list = []
    data =  {}
    comments = []
    with open(path,'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if line[0].startswith('#'): comments.append(line[0])
            data_list.append(line)
    #check length file using comments
    if len(comments) != len(set(comments)):
        indexes = []
        for i in range(len(data_list)):
            if comments[1] in data_list[i]:
                indexes.append(i)
        index_to_divide = indexes[-1] -1
    else:
        index_to_divide = 0
    for line in data_list[index_to_divide:]:
        if line[0].startswith('#'):continue
        data[line[0]] = line[1:]
    return data

def get_kos(path):
    ind_b = path.rfind('/')
    ind = path.index('.emapper') 
    genome = path[ind_b+1:ind]
    annotated_file = get_file(path)
    ko_list = []
    for query in annotated_file.keys():
        ko = annotated_file[query][10]
        if ko != '-' and ',' not in ko:
            ko_list.append(ko)
        elif ',' in ko:
            ko_list += ko.split(',')
    ko_list = [k[3:] for k in ko_list]
    genome_data_ko = {}
    for ko in ko_list:
        genome_data_ko[ko] = ko_list.count(ko)
    return genome, genome_data_ko, len(set(ko_list)) 

def get_validation_set(to_validate, training_set):
    training_kos = training_set.columns
    validation_kos = to_validate.columns
    if len(set(training_kos).intersection(set(validation_kos))) == 0:
        print('no common kos between provided ones and training')
        return
    else:
        common = list(set(training_kos).intersection(set(validation_kos)))
        print('{} common kos'.format(len(common)))
        # The `common_table` variable is created by selecting only the columns that are common between
        # the provided user dataset and the training dataset. It ensures that only the shared columns
        # (KOs) are included in the `common_table` for further processing. This step helps in aligning
        # the datasets properly for validation and prediction tasks.
        common_table = to_validate[common]
        #remove orthologs in validation not in the training
        to_remove = set(validation_kos) - set(training_kos)
        print('{} kos present in user set but not in training set will be removed'.format(len(to_remove)))
        missing = list(set(training_kos) - set(validation_kos))
        print('{} kos missing from the users et will be add to train the classifiers'.format(len(missing)))
        missed = pd.DataFrame(0, index = to_validate.index, columns= missing)
        to_submit = common_table.merge(missed, left_index = True, right_index = True)
        to_submit = to_submit[list(training_kos)] #change order columns
        print('Shape of training dataset: {}, Shape of user dataset: {}'.format(training_set.shape, to_submit.shape))
        print('\n')
        if list(to_submit.columns) != list(training_set.columns): 
            print('Error')
            return
    return to_submit

def get_tools_predictions(funcs, user_dataset, original_val_set):
    t_sets = {}
    models = {}
    scalers = {}
    vals_s = {}
    for f in funcs:
        if f in ad_classes and f != 'acetoclastic_methanogenesis':
            t_sets[f] = training_sets_dict[f][[k for k in list(training_sets_dict[f].columns) if k.startswith('K')]]
            models[f] = load(open("../saved_models/new_model_"+f+".sav", 'rb'))
            scalers[f] = load(open("../saved_models/new_scaler_"+f+".sav", 'rb'))
            vals_s[f] = get_validation_set(user_dataset, t_sets[f])
        elif f in ["anoxygenic_photoautotrophy_Fe_oxidizing","dark_sulfite_oxidation","oil_bioremediation","dark_sulfur_oxidation","dark_thiosulfate_oxidation","anoxygenic_photoautotrophy_S_oxidizing"]:
            t_sets[f] = original_training_dataset
            m = "../saved_models/"+f+".mdl_wts.hdf5"
            models[f] = load_model(m, compile = True, custom_objects={"matthews_correlation_coefficient": matthews_correlation_coefficient })
            scalers[f] = load(open("../saved_models/scaler_"+f+".sav", 'rb'))
            vals_s[f] = original_val_set
        elif f == 'acetoclastic_methanogenesis':
            t_sets[f] = training_sets_dict[f][[k for k in list(training_sets_dict[f].columns) if k.startswith('K')]]
            models[f] = load(open("../saved_models/new_model_"+f+"_2.sav", 'rb'))
            scalers[f] = load(open("../saved_models/new_scaler_"+f+"_2.sav", 'rb'))
            vals_s[f] = get_validation_set(user_dataset, t_sets[f])
        else:
            t_sets[f] = original_training_dataset
            models[f] = load(open("../saved_models/model_"+f+".sav", 'rb'))
            scalers[f] = load(open("../saved_models/scaler_"+f+".sav", 'rb'))
            vals_s[f] = original_val_set
    return t_sets, vals_s, models, scalers

def validate(t_vals, models, scalers, function_validation = 0):
    print('Predicting...')
    print('\n')
    results_per_class = {}
    scores = {}
    for c in list(t_vals.keys()):
        print(c)
        if c in ["anoxygenic_photoautotrophy_Fe_oxidizing","dark_sulfite_oxidation","oil_bioremediation","dark_sulfur_oxidation","dark_thiosulfate_oxidation","anoxygenic_photoautotrophy_S_oxidizing"]:
            to_validate_norm = scalers[c].transform(t_vals[c])
            pred1 = (models[c].predict(to_validate_norm) > 0.5).astype(np.int32)
            pred = []
            for i in pred1:
                pred.append(i[0])
        else:
            to_validate_norm = scalers[c].transform(t_vals[c])
            pred = models[c].predict(to_validate_norm)

        results_per_class[c] = pred
        
        if type(function_validation) != int:
            scores[c] = [matthews_corrcoef(function_validation[c], pred), f1_score(function_validation[c], pred, zero_division=1), confusion_matrix(function_validation[c], pred), accuracy_score(function_validation[c], pred), hamming_loss(function_validation[c], pred), zero_one_loss(function_validation[c], pred),  function_validation[c].sum()]
    
    print('\n')
    print('Done!')
    return results_per_class, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--user_dataset', required = True, help = '.csv file containing a matrix with KOs as columns and genomes as rows, the first column must be unnamed. The folder containing eggNOG-mapper annotation files can be provided as an alternative')
    parser.add_argument('-f', '--functions', help = 'matrix containing functions as columns and genomes as rows, used for checking the results of classifier')
    parser.add_argument('-ad', '--predict_AD_functions', action= 'store_true', help = 'include refined and new models for prediction of AD functions')
    parser.add_argument('-o', '--output_folder', required = True, help = 'path to EXISTING output directory')
    args = parser.parse_args()
    
    ###Functions to predict

    classes = ['anoxygenic_photoautotrophy_Fe_oxidizing', 'oil_bioremediation', 'dark_sulfite_oxidation', 'arsenate_respiration', 
               'manganese_respiration', 'dark_sulfur_oxidation', 'knallgas_bacteria', 'reductive_acetogenesis', 'dark_iron_oxidation', 
               'dark_thiosulfate_oxidation', 'chlorate_reducers', 'iron_respiration', 'anoxygenic_photoautotrophy_H2_oxidizing', 
               'nitrate_denitrification', 'chitinolysis', 'aerobic_anoxygenic_phototrophy', 'denitrification', 'dissimilatory_arsenate_reduction', 
               'dark_sulfide_oxidation', 'ureolysis', 'cellulolysis', 'thiosulfate_respiration', 'nitrous_oxide_denitrification', 
               'plastic_degradation', 'sulfur_respiration', 'aromatic_hydrocarbon_degradation', 'xylanolysis', 
               'sulfite_respiration', 'fumarate_respiration', 'dark_hydrogen_oxidation', 'nitrification', 'methanol_oxidation', 'sulfate_respiration',
                 'dark_oxidation_of_sulfur_compounds', 'nitrite_denitrification', 'arsenate_detoxification', 'anoxygenic_photoautotrophy_S_oxidizing',
                   'nitrate_respiration', 'nitrite_respiration', 'aromatic_compound_degradation', 'nitrate_ammonification', 'ligninolysis', 
                   'nitrite_ammonification', 'phototrophy', 'respiration_of_sulfur_compounds', 'anoxygenic_photoautotrophy', 'methylotrophy', 
                   'nitrogen_fixation', 'invertebrate_parasites', 'nitrogen_respiration', 'photoheterotrophy', 'chemoheterotrophy', 'nitrate_reduction'
                   , 'aerobic_ammonia_oxidation', 'predatory_or_exoparasitic', 'methanogenesis_using_formate', 'plant_pathogen', 
                   'human_pathogens_meningitis', 'human_pathogens_gastroenteritis', 'hydrocarbon_degradation', 'manganese_oxidation', 
                   'animal_parasites_or_symbionts', 'human_pathogens_all', 'photoautotrophy', 'human_pathogens_septicemia', 'aerobic_chemoheterotrophy',
                     'human_associated', 'aliphatic_non_methane_hydrocarbon_degradation', 'human_pathogens_pneumonia', 'fermentation', 
                     'human_pathogens_diarrhea', 'mammal_gut', 'methanotrophy', 'human_gut', 'intracellular_parasites', 'methanogenesis_by_CO2_reduction_with_H2',
                     'methanogenesis_by_disproportionation_of_methyl_groups', 'methanogenesis_by_reduction_of_methyl_compounds_with_H2', 
                     'hydrogenotrophic_methanogenesis', 'oxygenic_photoautotrophy', 'aerobic_nitrite_oxidation', 'methanogenesis', 
                     'arsenite_oxidation_detoxification', 'arsenite_oxidation_energy_yielding', 'fish_parasites', 'dissimilatory_arsenite_oxidation', 
                     'photosynthetic_cyanobacteria', 'human_pathogens_nosocomia'][::-1]

    #Extra AD functions, predicted only if --ad is specified as inputs
    ad_classes = ['butanoate_oxidizers', 'cellulose_degradation', 'CO_oxidored', 'CO_oxired_homoacetogenesis', 'h2_production',
                  'homoacetogenesis', 'propionate_oxidizers', 'starch_degradation', 'sulfate_reduction', 'acetoclastic_methanogenesis']

    #Read inputs files
    print('\n')
    print('Welcome in MICROPHERRET, tool for prediction of microbial phenotypes!')
    print('\n')
    print('Prepare training and user test before running the models... \n')
    if args.user_dataset.endswith('.csv'): 
        user_dataset = pd.read_csv(args.user_dataset).set_index('Unnamed: 0')
    else:
        ###get annotation matrix
        print('Getting annotation matrix from eggNOG-mapper annotations...')
        files = [os.path.join(args.user_dataset,i) for i in os.listdir(args.user_dataset)]
        data_ko = {}
        genomes_list = []
        ko_number = {}
        for f in files:
            if f.endswith('.annotations'):
                print(f'Processing {f} file...')
                genome, data_ko[genome], ko_number[genome] = get_kos(f)
                genomes_list.append(genome)
        df = pd.DataFrame(data_ko).T
        df.fillna(0, inplace = True)
        ko_df = pd.DataFrame(data = data_ko).T
    user_dataset = ko_df.fillna(0)
    user_dataset.to_csv(os.path.join(args.output_folder, 'user_dataset_KO_matrix.csv'))
    print('Done')
    print('\n')
    
    original_training_dataset = pd.read_csv('../matrix/genome_ko_all.csv').set_index('Genome').drop('Species', axis = 1)
    original_val_dataset = get_validation_set(user_dataset, original_training_dataset)
    #State functions to predict
    if args.predict_AD_functions:
        extra = ad_classes
        print('This will take some time, be patient. I am loading the desired training sets for refined AD functions prediction ')
        print('\n')
    else: 
        extra = ['acetoclastic_methanogenesis']
    to_predict = classes + extra
    training_sets_dict = {i[:i.index('_tset.csv')]:pd.read_csv(os.path.join('../matrix/Training_sets',i)).set_index('Unnamed: 0') for i in os.listdir('../matrix/Training_sets') if i[:i.index('_tset.csv')] in extra}
       
    t_sets, t_vals, models, scalers = get_tools_predictions(to_predict, user_dataset, original_val_dataset)    
    
    if args.functions:
        functions = pd.read_csv(args.functions).drop(['Unnamed: 0', 'scientific_name'], axis =1).set_index('genome_id')
        functions_val = functions.loc[list(set(functions.index).intersection(set(original_val_dataset.index)))]

        results_per_class, scores = validate(t_vals, models, scalers, functions_val)
        results_df = pd.DataFrame(results_per_class, index = original_val_dataset.index)
        results_df.to_csv(os.path.join(args.output_folder,'predict_functions_exact.csv'))
        sums = results_df.sum()
        sums.to_csv(os.path.join(args.output_folder,'predict_sum_exact.csv'))
        scores_df = pd.DataFrame(scores).T
        scores_df.columns = ['MCC', 'f1_score', 'confusion_matrix', 'accuracy', 'hamming_loss', 'zero_one_loss','genome_n']
        scores_df.to_csv(os.path.join(args.output_folder,'validation_scores.csv'))
    else:
        results_per_class, scores = validate(t_vals, models, scalers)
        results_df = pd.DataFrame(results_per_class, index = original_val_dataset.index)
        results_df.to_csv(os.path.join(args.output_folder,'predict_functions.csv'))
        sums = results_df.sum()
        sums.to_csv(os.path.join(args.output_folder,'predict_sum.csv'))
