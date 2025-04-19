import re
import numpy as np

from pprint import pprint

# File path
file_path = "/home/anderson/parameters_results/hard_eval/experiment_merged_eval_results.txt"

# Define regex patterns for extracting the required metrics
patterns = {
    "total_accuracy": re.compile(r"estimated accuracy \(window size 1\): ([\d\.]+)%"),
    "total_ar_condicionado": re.compile(r"appliance ar_condicionado: ([\d\.]+)%"),
    "total_chuveiro": re.compile(r"appliance chuveiro: ([\d\.]+)%"),
    "total_refrigerador": re.compile(r"appliance refrigerador: ([\d\.]+)%"),
    "f1score_ar_condicionado": re.compile(r"appliance ar_condicionado - precision: [\d\.]+%, recall: [\d\.]+%, f1score: ([\d\.]+)%"),
    "f1score_chuveiro": re.compile(r"appliance chuveiro - precision: [\d\.]+%, recall: [\d\.]+%, f1score: ([\d\.]+)%"),
    "f1score_refrigerador": re.compile(r"appliance refrigerador - precision: [\d\.]+%, recall: [\d\.]+%, f1score: ([\d\.]+)%"),
    
    "precision_ar_condicionado": re.compile(r"appliance ar_condicionado - precision: ([\d\.]+)%"),
    "precision_chuveiro": re.compile(r"appliance chuveiro - precision: ([\d\.]+)%"),
    "precision_refrigerador": re.compile(r"appliance refrigerador - precision: ([\d\.]+)%"),

    "recall_ar_condicionado": re.compile(r"appliance ar_condicionado - precision: [\d\.]+%, recall: ([\d\.]+)%"),
    "recall_chuveiro": re.compile(r"appliance chuveiro - precision: [\d\.]+%, recall: ([\d\.]+)%"),
    "recall_refrigerador": re.compile(r"appliance refrigerador - precision: [\d\.]+%, recall: ([\d\.]+)%"),
}

# Containers for metrics per subscenario
metrics = {
    "casa_diego": {key: [] for key in patterns},
    "casa_andrey": {key: [] for key in patterns},
}

# Read file and extract values
with open(file_path, "r") as file:
    current_subscenario = None
    for line in file:
        # Identify subscenario
        if "Subcenario:  casa_diego" in line:
            current_subscenario = "casa_diego"
        elif "Subcenario:  casa_andrey" in line:
            current_subscenario = "casa_andrey"
        elif "Subscenario: casa_tipo1" in line:
            current_subscenario = None
        
        # Extract metrics if in relevant subscenario
        if current_subscenario:
            for key, pattern in patterns.items():
                match = pattern.search(line)
                if match:
                    metrics[current_subscenario][key].append(np.float64(match.group(1)))

# Compute averages
averages = {
    scenario: {metric: np.round(np.mean(values), 2) for metric, values in data.items()}
    for scenario, data in metrics.items()
}

pprint(averages)
