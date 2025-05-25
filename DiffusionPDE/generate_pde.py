import yaml
from argparse import ArgumentParser
from scripts import generate_TE_heat, generate_NS_heat, generate_MHD, generate_E_flow, generate_VA, generate_Elder
import sys

if __name__ == "__main__":
    arg = sys.argv  # 接收参数

    parser = ArgumentParser(description='Generate PDE file')
    parser.add_argument('--config', type=str, help='Path to config file')
    options = parser.parse_args()


    config_path = options.config
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    name = config['data']['name']

    if name == 'TE_heat':
        print('Solving TE_heat equation...')
        generate_TE_heat(config)
    elif name == 'NS_heat':
        print('Solving NS_heat equation...')
        generate_NS_heat(config)

    elif name == 'MHD':
        print('Solving MHD equation...')
        generate_MHD(config)

    elif name == 'E_flow':
        print('Solving E_flow equation...')
        generate_E_flow(config)

    elif name == 'VA':
        print('Solving VA equation...')
        generate_VA(config)
        
    elif name == 'Elder':
        print('Solving Elder equation...')
        generate_Elder(config)

    else:
        print('PDE not found')
        exit(1)