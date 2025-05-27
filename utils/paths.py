from pathlib import Path

HOME_PATH = Path.home()

ROOT_PATH = Path(__file__).resolve().parents[1]
PLOTS_PATH = ROOT_PATH.joinpath("./plots")
RESULT_PATH = ROOT_PATH.joinpath("./results")
DATA_PATH = ROOT_PATH.joinpath("./data")
CONF_PATH = ROOT_PATH.joinpath("./conf")
