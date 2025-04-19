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

# Environment and path settings
CURRENT_SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_PATH="$(cd "$CURRENT_SCRIPT_PATH/.." && pwd)"
NIALM_PATH="$HOME/envs/nialm3"
SET_PATH="$HOME/envs/set-nialm3"
EVAL_BASE_PATH="$HOME/temp/individual_appliances/residencial/"

# Base directory for storing results
RESULTS_BASE_DIR="$HOME/parameters_results/$EVAL_TYPE"
mkdir -p "$RESULTS_BASE_DIR"

# Define the three augmentation method arguments.
# (They are passed to data_aug_methods.py as is.)
# augmentation_methods=("" --synthetic_modelling --random_assign --merged)
augmentation_methods=("")

# Loop over each augmentation method
for method in "${augmentation_methods[@]}"; do
    # Remove leading '--' for using in folder and file names
    method_name="${method#--}"
    method_name=${method_name:-"no_args"}
    echo "Processing augmentation method: $method_name"

    if [ "$method_name" == "no_args" ] ; then
        cp "$PROJECT_PATH/conf/residencial1_1.conf" "$HOME/envs/set-nialm3/configs/individual_appliances/residencial/residencial_default.conf"
    else
        cp "$PROJECT_PATH/conf/residencial1_2.conf" "$HOME/envs/set-nialm3/configs/individual_appliances/residencial/residencial_default.conf"
    fi

    # Create an aggregated eval result file for this method
    AGG_RESULT_FILE="${RESULTS_BASE_DIR}/experiment_${method_name}_eval_results.txt"
    # Empty (or create) the aggregated results file
    : > "$AGG_RESULT_FILE"

    # Copy appropriate config file
    if [ "$method_name" == "random_assign" ] || [ "$method_name" == "no_args" ] || [ "$method_name" == "signal_transform" ]; then
        cp "$PROJECT_PATH/conf/config1_1.json" "$HOME/envs/set-nialm3/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    else
        cp "$PROJECT_PATH/conf/config1_2.json" "$HOME/envs/set-nialm3/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    fi

    # Run three iterations for the current augmentation method
    for run in {1..3}; do
        echo "------------------------------"
        echo "Starting evaluation run $run for augmentation method '$method_name'"
        echo "------------------------------"

        EXPERIMENT_TAG="experiment_${method_name}_run${run}"
        EXPERIMENT_FOLDER="${RESULTS_BASE_DIR}/${EXPERIMENT_TAG}"
        EXPERIMENT_MODEL_CONFIG="${EXPERIMENT_FOLDER}/model"

        cp -r "$EXPERIMENT_MODEL_CONFIG/"* "$EVAL_BASE_PATH/"

        # Register each appliance for set-nialm3 module
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/ar_condicionado.conf)
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/refrigerador.conf)
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/chuveiro.conf)

        # Run the evaluation step and append the results to the aggregated result file.
        echo "#### RUN $run - $method_name ####" >> "$AGG_RESULT_FILE"
        (cd "$SET_PATH" && scripts/nialm_gen.sh -v --eval configs/individual_appliances/residencial/residencial.conf) >> "$AGG_RESULT_FILE"
        echo -e "\n#### RUN $run COMPLETED ####\n" >> "$AGG_RESULT_FILE"

        echo "Completed run $run for method '$method_name'. Results stored in $EXPERIMENT_FOLDER and appended to $AGG_RESULT_FILE"
    done
done

echo "Evaluation completed."
