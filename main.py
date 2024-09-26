import yaml
from utils.eval import eval_model, prepare_for_eval
from utils.utils import load_data, NumpyEncoder
import warnings
import json
warnings.filterwarnings('ignore')


with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


dataset_name = config["globals"]["dataset"]
data = load_data(dataset_name)

for model_name in config["models"]:

    if not config["models"][model_name]["eval"]:
        continue
    print("-"*50)
    print(f"\nProcessing {model_name}\n")
    print("-"*50)
    (learner, model, loader) = prepare_for_eval(config["models"][model_name])

    out = eval_model(learner, model, data,
                     config["models"][model_name],
                     loader, save_score=config["models"][model_name]["save_score"])

    output_path_1 = config["globals"]["output_folder"] + f"/{model_name}_{dataset_name}.json"
    with open(output_path_1, "w") as file:
        json.dump(out[0], file, indent=4, cls=NumpyEncoder)

    if config["models"][model_name]["save_score"]:
        output_path_2 = f"{config["globals"]["output_folder"]}/{model_name}_{dataset_name}_scores.json"
        with open(output_path_2, "w") as file:
            json.dump(out[1], file, indent=4, cls=NumpyEncoder)
