from src.main_helper import prepare_training
from src.train import train


def main():
    model, device, network_config, train_options, this_run_folder, tb_logger = prepare_training('hidden')
    train(model, device, train_options, this_run_folder, tb_logger)


if __name__ == '__main__':
    main()
