#!/usr/bin/python3

# Keep TensorFlow/Keras output quiet during normal command-line use.
# This must be set before importing TensorFlow or Keras.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import logging
import warnings
from pathlib import Path
from pickle import load

import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from keras.models import load_model
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
tf.get_logger().setLevel("ERROR")
logging.getLogger("tensorflow").setLevel(logging.ERROR)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hamming_loss,
    matthews_corrcoef,
    zero_one_loss,
)


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
    data = {}
    comments = []
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            if line[0].startswith('#'):
                comments.append(line[0])
            data_list.append(line)

    if len(comments) != len(set(comments)):
        indexes = []
        for i in range(len(data_list)):
            if comments[1] in data_list[i]:
                indexes.append(i)
        index_to_divide = indexes[-1] - 1
    else:
        index_to_divide = 0

    for line in data_list[index_to_divide:]:
        if line[0].startswith('#'):
            continue
        data[line[0]] = line[1:]
    return data


def get_kos(path):
    path = str(path)
    ind_b = path.rfind('/')
    ind = path.index('.emapper')
    genome = path[ind_b + 1:ind]
    annotated_file = get_file(path)
    ko_list = []

    for query in annotated_file:
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


def read_matrix(path, preferred_index=None):
    """Read a CSV matrix and use its identifier column as index."""
    df = pd.read_csv(path)

    if preferred_index and preferred_index in df.columns:
        return df.set_index(preferred_index)
    if 'Unnamed: 0' in df.columns:
        return df.set_index('Unnamed: 0')

    # Fall back to the first column, which is how MICROPHERRET matrices are stored.
    return df.set_index(df.columns[0])


def get_validation_set(to_validate, training_set, verbose=False):
    training_kos = list(training_set.columns)
    validation_kos = list(to_validate.columns)
    common = [ko for ko in training_kos if ko in to_validate.columns]

    if not common:
        raise ValueError('No common KOs between the provided dataset and the training set.')

    to_remove = set(validation_kos) - set(training_kos)
    missing = [ko for ko in training_kos if ko not in to_validate.columns]

    if verbose:
        print(f'{len(common)} common KOs')
        print(f'{len(to_remove)} KOs present in user set but not in training set will be removed')
        print(f'{len(missing)} KOs missing from the user set will be added with value 0')

    # reindex does all three operations at once: removes extra KOs, adds missing KOs as 0,
    # and restores exactly the training-column order.
    to_submit = to_validate.reindex(columns=training_kos, fill_value=0)

    if verbose:
        print(
            'Shape of training dataset: {}, Shape of user dataset: {}'.format(
                training_set.shape, to_submit.shape
            )
        )
        print()
    return to_submit


def load_training_sets(folder, functions):
    """Load <function>_tset.csv files for the requested functions only."""
    folder = Path(folder)
    training_sets = {}

    if not folder.is_dir():
        raise FileNotFoundError(f'Training-set directory not found: {folder}')

    for path in folder.glob('*_tset.csv'):
        function_name = path.name[:-len('_tset.csv')]
        if function_name in functions:
            training_sets[function_name] = read_matrix(path)

    missing = sorted(set(functions) - set(training_sets))
    if missing:
        raise FileNotFoundError(
            'Missing training-set files for: ' + ', '.join(missing) +
            f'. Expected them in {folder} as <function>_tset.csv.'
        )

    return training_sets


def prepare_models(functions, user_dataset, paths, ad_classes, verbose=False):
    t_vals = {}
    models = {}
    scalers = {}

    # All legacy MICROPHERRET classes use the same original training KO space,
    # except acetoclastic methanogenesis and the new AD models.
    legacy_functions = [
        f for f in functions
        if f not in ad_classes and f != 'acetoclastic_methanogenesis'
    ]

    original_training_dataset = None
    original_val_dataset = None
    if legacy_functions:
        original_training_dataset = read_matrix(paths['original_matrix'], preferred_index='Genome')
        if 'Species' in original_training_dataset.columns:
            original_training_dataset = original_training_dataset.drop(columns='Species')
        original_val_dataset = get_validation_set(user_dataset, original_training_dataset, verbose=verbose)

    ad_training_sets = {}
    requested_ad = [f for f in functions if f in ad_classes]
    if requested_ad:
        ad_training_sets = load_training_sets(paths['ad_training_sets'], requested_ad)

    aceto_training_sets = {}
    if 'acetoclastic_methanogenesis' in functions:
        aceto_training_sets = load_training_sets(
            paths['legacy_training_sets'], ['acetoclastic_methanogenesis']
        )

    keras_classes = {
        'anoxygenic_photoautotrophy_Fe_oxidizing',
        'dark_sulfite_oxidation',
        'oil_bioremediation',
        'dark_sulfur_oxidation',
        'dark_thiosulfate_oxidation',
        'anoxygenic_photoautotrophy_S_oxidizing',
    }

    for f in tqdm(functions, desc='Loading models', unit='model'):
        if f in ad_classes:
            training = ad_training_sets[f]
            training = training[[c for c in training.columns if str(c).startswith('K')]]
            t_vals[f] = get_validation_set(user_dataset, training, verbose=verbose)
            models[f] = load(open(paths['new_models'] / f'model_{f}.sav', 'rb'))
            scalers[f] = load(open(paths['new_models'] / f'scaler_{f}.sav', 'rb'))

        elif f == 'acetoclastic_methanogenesis':
            training = aceto_training_sets[f]
            training = training[[c for c in training.columns if str(c).startswith('K')]]
            t_vals[f] = get_validation_set(user_dataset, training, verbose=verbose)
            models[f] = load(open(paths['saved_models'] / 'new_model_acetoclastic_methanogenesis_2.sav', 'rb'))
            scalers[f] = load(open(paths['saved_models'] / 'new_scaler_acetoclastic_methanogenesis_2.sav', 'rb'))

        elif f in keras_classes:
            t_vals[f] = original_val_dataset
            models[f] = load_model(
                paths['saved_models'] / f'{f}.mdl_wts.hdf5',
                compile=True,
                custom_objects={
                    'matthews_correlation_coefficient': matthews_correlation_coefficient
                },
            )
            scalers[f] = load(open(paths['saved_models'] / f'scaler_{f}.sav', 'rb'))

        else:
            t_vals[f] = original_val_dataset
            models[f] = load(open(paths['saved_models'] / f'model_{f}.sav', 'rb'))
            scalers[f] = load(open(paths['saved_models'] / f'scaler_{f}.sav', 'rb'))

    return t_vals, models, scalers


def validate(t_vals, models, scalers, function_validation=None):
    print('Predicting...\n')
    results_per_class = {}
    scores = {}

    keras_classes = {
        'anoxygenic_photoautotrophy_Fe_oxidizing',
        'dark_sulfite_oxidation',
        'oil_bioremediation',
        'dark_sulfur_oxidation',
        'dark_thiosulfate_oxidation',
        'anoxygenic_photoautotrophy_S_oxidizing',
    }

    for c, validation_matrix in tqdm(
        t_vals.items(),
        total=len(t_vals),
        desc='Predicting phenotypes',
        unit='model',
    ):
        to_validate_norm = scalers[c].transform(validation_matrix)

        if c in keras_classes:
            raw = models[c].predict(to_validate_norm, verbose=0)
            pred = (raw > 0.5).astype(np.int32).reshape(-1)
        else:
            pred = np.asarray(models[c].predict(to_validate_norm)).reshape(-1)

        results_per_class[c] = pred

        if function_validation is not None and c in function_validation.columns:
            y_true = function_validation[c].reindex(validation_matrix.index)
            valid = y_true.notna()

            if valid.any():
                y_true_valid = y_true.loc[valid].astype(int).to_numpy()
                y_pred_valid = pd.Series(pred, index=validation_matrix.index).loc[valid].astype(int).to_numpy()

                scores[c] = [
                    matthews_corrcoef(y_true_valid, y_pred_valid),
                    f1_score(y_true_valid, y_pred_valid, zero_division=1),
                    confusion_matrix(y_true_valid, y_pred_valid),
                    accuracy_score(y_true_valid, y_pred_valid),
                    hamming_loss(y_true_valid, y_pred_valid),
                    zero_one_loss(y_true_valid, y_pred_valid),
                    int(y_true_valid.sum()),
                    int(len(y_true_valid)),
                ]

    return results_per_class, scores


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--user_dataset', required=True,
        help='.csv KO matrix (genomes as rows, KOs as columns) or a folder containing eggNOG-mapper .annotations files'
    )
    parser.add_argument(
        '-f', '--functions',
        help='optional CSV containing known phenotype/function labels, used to validate predictions'
    )
    parser.add_argument(
        '-ad', '--predict_AD_functions', action='store_true',
        help='also predict the 9 new/refined anaerobic-digestion functions'
    )
    parser.add_argument(
        '-o', '--output_folder', required=True,
        help='path to an existing output directory'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='print additional preprocessing details'
    )
    args = parser.parse_args()

    standard_classes = ['human_pathogens_nosocomia', 'photosynthetic_cyanobacteria', 'dissimilatory_arsenite_oxidation', 'fish_parasites', 'arsenite_oxidation_energy_yielding', 'arsenite_oxidation_detoxification', 'methanogenesis', 'aerobic_nitrite_oxidation', 'oxygenic_photoautotrophy', 'hydrogenotrophic_methanogenesis', 'methanogenesis_by_reduction_of_methyl_compounds_with_H2', 'methanogenesis_by_disproportionation_of_methyl_groups', 'methanogenesis_by_CO2_reduction_with_H2', 'intracellular_parasites', 'human_gut', 'methanotrophy', 'mammal_gut', 'human_pathogens_diarrhea', 'fermentation', 'human_pathogens_pneumonia', 'aliphatic_non_methane_hydrocarbon_degradation', 'human_associated', 'aerobic_chemoheterotrophy', 'human_pathogens_septicemia', 'photoautotrophy', 'human_pathogens_all', 'animal_parasites_or_symbionts', 'manganese_oxidation', 'hydrocarbon_degradation', 'human_pathogens_gastroenteritis', 'human_pathogens_meningitis', 'plant_pathogen', 'methanogenesis_using_formate', 'predatory_or_exoparasitic', 'aerobic_ammonia_oxidation', 'nitrate_reduction', 'chemoheterotrophy', 'photoheterotrophy', 'nitrogen_respiration', 'invertebrate_parasites', 'nitrogen_fixation', 'methylotrophy', 'anoxygenic_photoautotrophy', 'respiration_of_sulfur_compounds', 'phototrophy', 'nitrite_ammonification', 'ligninolysis', 'nitrate_ammonification', 'aromatic_compound_degradation', 'nitrite_respiration', 'nitrate_respiration', 'anoxygenic_photoautotrophy_S_oxidizing', 'arsenate_detoxification', 'nitrite_denitrification', 'dark_oxidation_of_sulfur_compounds', 'sulfate_respiration', 'methanol_oxidation', 'nitrification', 'dark_hydrogen_oxidation', 'fumarate_respiration', 'sulfite_respiration', 'xylanolysis', 'aromatic_hydrocarbon_degradation', 'sulfur_respiration', 'plastic_degradation', 'nitrous_oxide_denitrification', 'thiosulfate_respiration', 'cellulolysis', 'ureolysis', 'dark_sulfide_oxidation', 'dissimilatory_arsenate_reduction', 'denitrification', 'aerobic_anoxygenic_phototrophy', 'chitinolysis', 'nitrate_denitrification', 'anoxygenic_photoautotrophy_H2_oxidizing', 'iron_respiration', 'chlorate_reducers', 'dark_thiosulfate_oxidation', 'dark_iron_oxidation', 'reductive_acetogenesis', 'knallgas_bacteria', 'dark_sulfur_oxidation', 'manganese_respiration', 'arsenate_respiration', 'dark_sulfite_oxidation', 'oil_bioremediation', 'anoxygenic_photoautotrophy_Fe_oxidizing']
    # The 89th original MICROPHERRET phenotype uses its own training set/model.
    standard_classes = standard_classes + ['acetoclastic_methanogenesis']

    ad_classes = [
        'butyrate_oxidation',
        'cellulose_degradation',
        'starch_degradation',
        'propionate_oxidation',
        'h2_production',
        'homoacetogenesis',
        'wlp',
        'co_oxidoreduction',
        'sulfate_reduction',
    ]

    output_folder = Path(args.output_folder)
    if not output_folder.is_dir():
        raise NotADirectoryError(f'Output directory does not exist: {output_folder}')

    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent
    paths = {
        'original_matrix': project_dir / 'matrix' / 'genome_ko_all.csv',
        'legacy_training_sets': project_dir / 'matrix' / 'Training_sets',
        'ad_training_sets': project_dir / 'matrix' / 'Training_sets' / 'new_sets',
        'saved_models': project_dir / 'saved_models',
        'new_models': project_dir / 'saved_models' / 'new_models',
    }

    print('\nWelcome in MICROPHERRET, tool for prediction of microbial phenotypes!\n')
    print('Preparing user dataset...')
    if args.predict_AD_functions:
        print('This might take some time, be patient.')
    print()

    if args.user_dataset.endswith('.csv'):
        user_dataset = read_matrix(args.user_dataset)
    else:
        annotation_dir = Path(args.user_dataset)
        if not annotation_dir.is_dir():
            raise NotADirectoryError(f'Input is neither a CSV nor a directory: {annotation_dir}')

        if args.verbose:
            print('Getting annotation matrix from eggNOG-mapper annotations...')
        data_ko = {}
        for f in sorted(annotation_dir.iterdir()):
            if f.name.endswith('.annotations'):
                if args.verbose:
                    print(f'Processing {f} file...')
                genome, genome_kos, _ = get_kos(f)
                data_ko[genome] = genome_kos

        if not data_ko:
            raise FileNotFoundError(
                f'No files ending in .annotations were found in {annotation_dir}'
            )

        user_dataset = pd.DataFrame(data_ko).T.fillna(0)
        user_dataset.to_csv(output_folder / 'user_dataset_KO_matrix.csv')

    print('Done\n')

    to_predict = list(standard_classes)
    if args.predict_AD_functions:
        to_predict += ad_classes

    t_vals, models, scalers = prepare_models(
        to_predict, user_dataset, paths, set(ad_classes), verbose=args.verbose
    )

    functions = None
    if args.functions:
        functions = read_matrix(args.functions)

        # Support the older validation-table layout: genome_id plus metadata columns.
        if 'genome_id' in functions.columns:
            functions = functions.set_index('genome_id')
        functions = functions.drop(
            columns=[c for c in ['scientific_name'] if c in functions.columns],
            errors='ignore'
        )

    results_per_class, scores = validate(t_vals, models, scalers, functions)

    results_df = pd.DataFrame(results_per_class, index=user_dataset.index)
    suffix = '_exact' if args.functions else ''
    results_df.to_csv(output_folder / f'predict_functions{suffix}.csv')
    results_df.sum().to_csv(output_folder / f'predict_sum{suffix}.csv')

    if args.functions:
        if scores:
            scores_df = pd.DataFrame.from_dict(
                scores,
                orient='index',
                columns=[
                    'MCC', 'f1_score', 'confusion_matrix', 'accuracy',
                    'hamming_loss', 'zero_one_loss', 'positive_n', 'evaluated_n'
                ],
            )
            scores_df.to_csv(output_folder / 'validation_scores.csv')
        else:
            print(
                'Warning: -f was provided, but none of its phenotype columns matched '
                'the functions predicted in this run.'
            )
    print(f'\nDone! Results saved in: {output_folder}')

