Scripts pour uni2ts:

* 1st: Clone git 
```bash
git clone https://github.com/SalesforceAIResearch/uni2ts.git

cd uni2ts

pip install -e '.[notebook]'
```

* 2nd: copy ``anom_utils.py``, ``eval_moirai.py`` & ``utils/metrics.py`` and paste them in the repository

* 3rd: run:

```bash
python eval_moirai.py
```

parameters :

* --path: path to the json file of the data
* --dataset_name: name of the dataset (it will just change the names of the output files)
* --seq_len: size of the context length for the model
* --seq_len_windows: size of the lengths when evaluating the model window-wise, default to 1
* --batch_size: batch size for the model
* --save_scores: to save scores or not