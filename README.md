# moving_mahalanobis
Github repo for moving mahalanobis model

# Data

The scripts read the data in JSON format. The data structure is as follows:
```
{
  "serie_name_1": {
    "X": [list containing the values of the series],
    "y": [list containing the labels of the series (1 for anomaly, 0 otherwise)]
  },
  "serie_name_2": ...
}
```

## Instructions for Processing Data (Non-JSON Format)

To handle data that is not in JSON format, follow the steps below:

1. **Download Data**  
   Download the data from the various datasets.

2. **Organize Data**  
   For each dataset, place the data files in the corresponding folder.

3. **Run the Data Loader Script**  
   Execute the following command in your terminal:

   ```bash
   python ./{dataset}_anomaly/{dataset}_loader.py

# Models

To run the model, choose the parameters on the file ``config.yaml`` and then run:
    ```
    pip install -e .

    python main.py


To run the other models, refer to the ``scripts_{model}`` folders and follow the instructions in the ``README.md`` file.

# Print results

When you run the script ``main.py``, the results and the scores (optional) will be store on the folder ``results``. Use:

    ```bash
    python print_results.py --path [path to json file of the results] --n_decimals [2]

to display the scores on the consol. You can eval the model with another window size by running:

    ```bash
    python eval_from_scores.py --path [path to json file of the scores] --seq_len [custom sequence length]