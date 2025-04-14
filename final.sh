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

# Environment and path settings
NIALM_PATH="$HOME/envs/nialm3"
SET_PATH="$HOME/envs/set-nialm3"

# Base directory for storing results
RESULTS_BASE_DIR="$HOME/parameters_results"
mkdir -p "$RESULTS_BASE_DIR"

# Define the three augmentation method arguments.
# (They are passed to data_aug_methods.py as is.)
# augmentation_methods=("" --synthetic_modelling --random_assign --merged)
augmentation_methods=("" --synthetic_modelling --random_assign)

# Loop over each augmentation method
for method in "${augmentation_methods[@]}"; do
    # Remove leading '--' for using in folder and file names
    method_name="${method#--}"
    method_name=${method_name:-"no_args"}
    echo "Processing augmentation method: $method_name"

    if [ "$method_name" == "no_args" ] ; then
        cp "$HOME/code_projects/mestrado/conf/residencial1_1.conf" "$HOME/envs/set-nialm3/configs/individual_appliances/residencial/residencial_default.conf"
    else
        cp "$HOME/code_projects/mestrado/conf/residencial1_2.conf" "$HOME/envs/set-nialm3/configs/individual_appliances/residencial/residencial_default.conf"
    fi

    # Create an aggregated eval result file for this method
    AGG_RESULT_FILE="${RESULTS_BASE_DIR}/experiment_${method_name}_eval_results.txt"
    # Empty (or create) the aggregated results file
    : > "$AGG_RESULT_FILE"

    # Copy appropriate config file
    if [ "$method_name" == "random_assign" ] || [ "$method_name" == "no_args" ] || [ "$method_name" == "signal_transform" ]; then
        cp "$HOME/code_projects/mestrado/conf/config1.json" "$HOME/envs/set-nialm3/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    else
        cp "$HOME/code_projects/mestrado/conf/config2.json" "$HOME/envs/set-nialm3/keras_disaggregators/tests/individual_appliances/residencial/residencial/config.json"
    fi

    # Run three iterations for the current augmentation method
    for run in {1..3}; do
        echo "------------------------------"
        echo "Starting run $run for augmentation method '$method_name'"
        echo "------------------------------"
        
        # Activate environment, run the data augmentation method script,
        # then deactivate the environment.
        nialm_env
        if [ -z "$method" ]; then
            python3 "data_aug_methods.py"
        else
            python3 "data_aug_methods.py" "$method"
        fi
        deactivate_env

        # Define a unique experiment folder for this run.
        # This folder will store the output files (dat, model, spec) from this run.
        EXPERIMENT_TAG="experiment_${method_name}_run${run}"
        EXPERIMENT_FOLDER="${RESULTS_BASE_DIR}/${EXPERIMENT_TAG}"
        mkdir -p "$EXPERIMENT_FOLDER/dat" "$EXPERIMENT_FOLDER/model" "$EXPERIMENT_FOLDER/spec"

        # Run training for each individual appliance configuration
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/ar_condicionado.conf)
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/refrigerador.conf)
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --train "$EXPERIMENT_TAG" configs/individual_appliances/residencial/chuveiro.conf)

        # Register each appliance (if registration is needed before evaluation)
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/ar_condicionado.conf)
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/refrigerador.conf)
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --register configs/individual_appliances/residencial/chuveiro.conf)

        # Run the evaluation step and append the results to the aggregated result file.
        echo "#### RUN $run - $method_name ####" >> "$AGG_RESULT_FILE"
        (cd "$SET_PATH" && \
         scripts/nialm_gen.sh -v --eval configs/individual_appliances/residencial/residencial.conf) >> "$AGG_RESULT_FILE"

        # Copy the output files from the temporary folder into the unique experiment folder.
        cp -r "$HOME/temp/"*.dat "$EXPERIMENT_FOLDER/dat/"
        cp -r "$HOME/temp/individual_appliances/residencial/"*"/" "$EXPERIMENT_FOLDER/model/"
        cp -r "$HOME/temp/individual_appliances/residencial/"*.dat "$EXPERIMENT_FOLDER/spec/"

        echo "Completed run $run for method '$method_name'. Results stored in $EXPERIMENT_FOLDER and appended to $AGG_RESULT_FILE"
    done
done

echo "All runs completed."
