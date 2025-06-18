# MICROPHERRET
MICROPHERRET: **MICRO**bial **PHE**notypic t**R**ait classifie**R** using machine l**E**arning **T**echniques

MICROPHERRET is a machine learning-based for the prediction of 89 metabolic and ecological prokaryotic phenotypes from gene annotations. Starting from KEGG Orthologs, it can efficiently classify genomes and metagenomes by integrating different supervised optimized machine learning models - logistic regression, random forest, support vector machines and neural networks - maximizing the prediction accuracy. MICROPHERRET is an efficient approach for functional prediction of MAGs till 70% of genome completeness. The tool can be easily expanded and refined by the users to predict functions of interest.

## Installation

Create a conda environment with Python 3.9 and packages:
```
conda create -n micropherret python=3.9 pandas biopython scikit-learn=1.1.2 numpy tensorflow
pip install tensorflow-addons
```
To enter the environment:
```
conda activate micropherret
```
Clone this git repository
```
git clone https://github.com/BizzoTL/MICROPHERRET/
```
The files needed to run the scripts can be retrieved here:  https://mega.nz/folder/FAVSRYTT#ElNlwvpSMuXSZu9NDoIp3w

To run the scripts, create a folder called "MICROPHERRET" and add the files and script into it.

## Requirements

Random forest models were saved in .sav files with the parameter n_job set to 6 or fewer. Consequently, training MICROPHERRET requires at least 6 CPUs. To modify this parameter, users can retrain and optimize the models by following the previously described procedure. 

## Usage
### Functional prediction

For simply predicting the functions of your genome of interest you need to provide the .annotations file from EggNOG-mapper. In the MICROPHERRET folder, put the annotations of your genomes or MAGs in a folder called "annotations". Run "get_annotation_matrix.ipynb" and specify the folder where the files are located. The obtained output, called "ko_df_all.csv", stores a matrix with the genetic information (KO copy number) per genome, with genomes as rows and KO as columns.

Here's an example of ko_df_all.csv's structure from "get_annotation_matrix.ipynb":

|         -         | KO<sub>1<sub> | KO<sub>2<sub> | ... | KO<sub>3<sub> |
| ----------------  | ------------- | ------------- | --- | ------------- |
| GCF_XXXXXXXXXX.X  |       2       |       0       | ... |       0       | 
| GCF_XXXXXXXXXX.X  |       0       |       1       | ... |       5       | 
|        ...        |      ...      |      ...      | ... |      ...      | 
| GCF_XXXXXXXXXX.X  |       1       |       3       | ... |       2       |


Run "predict_functions.py" from the "annotation" folder and provide the user-generated annotation matrix ko_df_all.csv as input to the script (```-i file.csv```). Additionally, a matrix containing real functions as columns and genomes as rows (e.g. function_db.csv) can be provided using the optional parameter ```-f file.csv``` to evaluate of prediction results. 

Example:
```
python predict_functions.py -i ko_df_all.csv -f function_db.csv
```

The resulting predict_function.csv file contains the genomes used in input as rows and the prediction for each functional class of MICROPHERRET in the columns. If the prediction is positive and the genome performs a particular function, a "1" is present in the cell, otherwise a "0". Here's an example:

|         -         | hydrocarbon_degradation | methanogenesis | ... | human_gut |
| ----------------  | ----------------------- | -------------- | --- | --------- |
| GCF_XXXXXXXXXX.X  |            0            |        1       | ... |     1     | 
| GCF_XXXXXXXXXX.X  |            1            |        1       | ... |     0     | 
|        ...        |           ...           |       ...      | ... |    ...    | 
| GCF_XXXXXXXXXX.X  |            0            |        0       | ... |     1     |

The number of genomes classified into each function is also saved in the file predict_sum.csv. The script predicts the functions of 1 genome/second.

### Class generation and refinement

MICROPHERRET's models can be refined to fit the user's needs by changing the sets of genomes performing the functions stored in the training set. At the same time, the training set can be expanded by modifying the training set adding genomes linked to the function the user wishes to predict. In both cases, the users can follow the instructions in "create_new_class.py". The inputs required to run the script are .annotations files from EggNOG-mapper for all the genomes linked to the function of interest. The script will then open and parse .annotations files and from each file it retrieves the KOs. Those are stored in a dictionary with KOs as keys and copy numbers as values. The dictionary is then used with the main script KO matrix to train the different machine-learning algorithms and after selecting the best-performing model it saves it. The resulting model and scaler can then be added to the folder containing MICROPHERRET models.

Parameters to use:
- ```-g folder```
  
 Folder containing the .annotations files from EggNOG-mapper of the genomes that will be used for model training

- ```--from_faprotax file.csv```
  
Dataset of genomes from FAPROTAX used for MICROPHERRET training, i.e. genome_ko_all.csv from the matrix folder

- ```-f string```
  
Name of the function-specific classifier that will be trained with the genomes. If it is new, a new model will be trained. Otherwise, a previous model will be retrained using the provided genomes 

- ```-r file.txt```
  
*optional* .txt file with the list of genomes that we want to remove from the previous training set, one per line. If genomes present in FAPROTAX are suspected to perform the function but were not provided in -g, they should be absolutely removed


Example:
```python create_new_class.py -g genomes_folder/ --from_faprotax faprotax_annotation_matrix.csv -f new_function -r genomes_to_remove.txt```

After creating the class, add the saved model and scaler with the others in the MICROPHERRET folder to run your predictions.

#### Class refinement example: acetoclastic methanogenesis

The code to run the script is available with the other files in the "Example" folder. After installing MICROPHERRET and activating the environment, simply run create_new_class.py from the folder to obtain a refined version of the classifier to predict acetoclastic methanogenesis. The exact command to run is the following:
```
python3 create_new_class.py -g annotations/ --from_faprotax my_faprotax_db.csv  -f acetoclastic_methanogenesis 
```
The resulting files will be the .sav and .scaler models that need to be put into the model folder in order to predict MICROPHERRET functions.

## Citation

If you use MICROPHERRET, please cite:

Bizzotto, E., Fraulini, S., Zampieri, G. et al. MICROPHERRET: MICRObial PHEnotypic tRait ClassifieR using Machine lEarning Techniques. Environmental Microbiome 19, 58 (2024). https://doi.org/10.1186/s40793-024-00600-6

