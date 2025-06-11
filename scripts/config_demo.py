from utils.config_loader import Config
from utils.args_parser import ArgsParser

def main():
    parser = ArgsParser()
    config_path, overrides = parser.parse_config_args()
    config = Config.from_file(config_path, overrides)

    print("=== Config ===")
    print(config)
    print("=== Model Config ===")
    print(config.model_config_typed)
    print("=== Flat Config ===")
    print(config.flat_config)
    print("=== Run ID ===")
    print(config.run_id)

if __name__ == "__main__":
    main()