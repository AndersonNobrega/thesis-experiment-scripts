#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

python3 utils/plot_pred.py --house "andrey"
python3 utils/plot_pred.py --house "diego"

python3 utils/plot_pred.py --house "andrey" --simple_eval
python3 utils/plot_pred.py --house "anderson" --simple_eval
python3 utils/plot_pred.py --house "diego" --simple_eval
python3 utils/plot_pred.py --house "igor" --simple_eval
python3 utils/plot_pred.py --house "leandro" --simple_eval

python3 utils/plot_papers_tendencia.py

python3 utils/plot_papers_distribuicao.py

python3 utils/plot_density.py

python3 utils/plot_overlapping_density.py

python3 utils/plot_histogram.py

python3 utils/plot_line_accuracy.py

python3 utils/plot_line_loss.py