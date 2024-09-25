import json
import numpy as np
from tqdm import tqdm
from utils.utils import create_windows


def eval_model(learner, model, data, params, loader, save_score=False):
    """
    Évalue un modèle sur un ensemble de données donné et retourne les résultats.

    Paramètres :
    -----------
    learner : object
        L'evaluator utilisé pour entraîner et évaluer le modèle, il comprends une fonctions get_params et fit().
    
    model : object
        Le modèle (exemple TS2Vec, Lagllama), si le modèle est deja compris dans l'evaluator, rentrer None
    
    data : dict
        Dictionnaire contenant les données d'entrée. Chaque clé du dictionnaire représente un ensemble de données, 
        et la valeur est un autre dictionnaire contenant 'X' et 'y' .
    
    params : dict
        Dictionnaire de paramètres à passer à l'évaluateur et au modèle. Il doit contenir :
        - 'seq_len' : longueur des séquences de données à traiter.
        - 'device' : appareil (CPU ou GPU) sur lequel exécuter le modèle.
        - 'model_name' : nom du modèle utilisé pour l'évaluation.
        - et paramètres spécifiques à chaque modèles
    
    loader : function
        Fonction ou classe qui transforme les données d'entrée 'X' en un format approprié pour l'évaluation 
        (par exemple, un DataLoader).
    
    save_score : bool, optionnel (par défaut=False)
        Si True, les scores et les labels pour chaque ensemble de données seront enregistrés dans un dictionnaire 
        et retournés avec les résultats.
    
    Retour :
    -------
    list_res : dict
        Dictionnaire contenant les résultats de l'évaluation pour chaque clé de données.
    
    list_scores : dict
        Dictionnaire contenant les scores et les labels pour chaque clé de données, uniquement si `save_score=True`.
    """    
    list_res = {}

    list_scores = {}

    for key in tqdm(data.keys()):
        X = data[key]["X"]
        y = data[key]["y"]

        dataset = loader(X, model, seq_len=params["seq_len"],
                         stride=params["seq_len"], device=params["device"])
        y = create_windows(y, seq_len=params["seq_len"], stride=params["seq_len"])
        evaluator = learner(y, dataset, model_name=params["model_name"], 
                            X=X, seq_len= params["seq_len"])
        evaluator.set_params(**params)
        list_res[key] = evaluator.fit()

        if save_score:
            list_scores[key] = {"scores": evaluator.score,
                                "labels": evaluator.labels}

    return list_res, list_scores
