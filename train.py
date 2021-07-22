import utils
from dataset import BraTS
from models import TransBTS

def main(config):
    


if __name__ == "__main__":
    args, unknown = utils.config.parse_args()
    config = utils.config.load_config(args, unknown)
    main_function(config)