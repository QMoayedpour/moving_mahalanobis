import json
import numpy as np
from tqdm import tqdm
import torch
from utils.utils import create_windows, useless
from model.ts2vec.ts2vec import TS2Vec_Learner
from model.ts2vec.utils import create_ts2vec_dataset
from model.M_mahala.movingmahala import MovingMahalanobis
from model.lagllama.utils.lagllama_loader import create_lagllama_dataset
from model.donut.donut import Donut


def eval_model(learner, model, data, params, loader):
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
        dataset = loader(x=X, model=model, seq_len=params["seq_len"],
                         stride=params["seq_len"], device=params["device"])
        #y = create_windows(y, seq_len=params["seq_len"], stride=params["seq_len"])
        evaluator = learner(X=X, y=y, dataloader=dataset, **params)
        #evaluator.set_params(**params)
        list_res[key] = evaluator.fit()

        if params["save_score"]:
            list_scores[key] = {"scores": evaluator.score,
                                "labels": evaluator.labels}

    return list_res, list_scores


def prepare_for_eval(config):

    if config["model_name"] == "TS2Vec":
        learner = MovingMahalanobis

        loader = create_ts2vec_dataset

        model = TS2Vec_Learner(
                input_dims=1,
                output_dims=config["n_channels"],
                train_data=np.random.randn(1, 1, 1),
                device=config["device"],
            )
        model.net.load_state_dict(torch.load(config["model_path"],
                                             map_location=torch.device(config["device"])))

    elif config["model_name"] == "lagllama_mahala":

        learner = MovingMahalanobis

        loader = create_lagllama_dataset

        model = config["model_path"]

    elif config["model_name"] == "Donut":
        learner = Donut

        loader = useless

        model = useless

    else:
        raise ValueError("No valids model names")

    return (learner, model, loader)
