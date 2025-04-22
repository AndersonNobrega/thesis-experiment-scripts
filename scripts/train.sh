#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Functions to activate and deactivate the environment
nialm_env() {
    echo "Changing to Nialm env"
    source "$NIALM_PATH/bin/activate"
}

deactivate_env() {
    deactivate
}

EVAL_TYPE="hard_eval"  # default
if [[ "$1" == "--simple_eval" ]]; then
    EVAL_TYPE="simple_eval"
elif [[ "$1" == "--hard_eval" ]]; then
    EVAL_TYPE="hard_eval"
fi

# Set config filenames based on EVAL_TYPE
if [[ "$EVAL_TYPE" == "hard_eval" ]]; then
    RESIDENCIAL_CONF_1="residencial1_1.conf"
    RESIDENCIAL_CONF_2="residencial1_2.conf"
    CONFIG_JSON_1="config1_1.json"
    CONFIG_JSON_2="config1_2.json"
else
    RESIDENCIAL_CONF_1="residencial2_1.conf"
    RESIDENCIAL_CONF_2="residencial2_2.conf"
    CONFIG_JSON_1="config2_1.json"
    CONFIG_JSON_2="config2_2.json"
fi

# Environment and path settings
CURRENT_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$(cd "$CURRENT_SCRIPT_PATH/.." && pwd)"
NIALM_PATH="$HOME/envs/nialm3"
SET_PATH="$HOME/envs/set-nialm3"
EVAL_BASE_PATH="$HOME/temp/individual_appliances/residencial/"

# Base directory for storing results
RESULTS_BASE_DIR="$PROJECT_PATH/results/$EVAL_TYPE"
mkdir -p "$RESULTS_BASE_DIR"

# Define the three augmentation method arguments.
# (They are passed to data_aug_methods.py as is.)
augmentation_methods=("" --synthetic_modelling --random_assign --merged)

# Loop over each augmentation method
for method in "${augmentation_methods[@]}"; do
    # Remove leading '--' for using in folder and file names
    method_name="${method#--}"
    method_name=${method_name:-"no_args"}
    echo "Processing augmentation method: $method_name"

    # Select correct residencial config
    if [ "$method_name" == "no_args" ]; then
        cp "$PROJECT_PATH/conf/$RESIDENCIAL_CONF_1" "$SET_PATH/configs/individual_appliances/residencial/residencial_default.conf"
    else
        cp "$PROJECT_PATH/conf/$RESIDENCIAL_CONF_2" "$SET_PATH/configs/individual_appliances/residencial/residencial_default.conf"
    fi

    # Copy appropriate config file
    if [ "$method_name" == "random_assign" ] || [ "$method_name" == "no_args" ] || [ "$method_name" == "signal_transform" ]; then
        cp "$PROJECT_PATH/conf/$CONFIG_JSON_1" "$SET_PATH/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    else
        cp "$PROJECT_PATH/conf/$CONFIG_JSON_2" "$SET_PATH/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    fi

    # Run three iterations for the current augmentation method
    for run in {1..3}; do
        echo "------------------------------"
        echo "Starting evaluation run $run for augmentation method '$method_name'"
        echo "------------------------------"

        # Run the data augmentation method script.
        nialm_env
        if [ -z "$method" ]; then
            python3 "$PROJECT_PATH/data_aug.py" "--$EVAL_TYPE"
        else
            python3 "$PROJECT_PATH/data_aug.py" "--$method_name" "--$EVAL_TYPE"
        fi
        deactivate_env

        EXPERIMENT_TAG="experiment_${method_name}_run${run}"
        EXPERIMENT_FOLDER="${RESULTS_BASE_DIR}/${EXPERIMENT_TAG}"
        mkdir -p "$EXPERIMENT_FOLDER/dat" "$EXPERIMENT_FOLDER/model" "$EXPERIMENT_FOLDER/spec"

        # Run training for each individual appliance configuration
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/ar_condicionado.conf)
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/refrigerador.conf)
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/chuveiro.conf)

        cp -r "$HOME/temp/individual_appliances/residencial/"*"/" "$EXPERIMENT_FOLDER/model/"
        cp -r "$HOME/temp/individual_appliances/residencial/"*.dat "$EXPERIMENT_FOLDER/spec/"

        echo "Completed training run $run for method '$method_name'. Results stored in $EXPERIMENT_FOLDER"
    done
done

echo "Training completed."
