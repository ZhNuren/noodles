import yaml
import subprocess

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    return config

def write_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file)

def update_yaml(config, model, path):
    config['MODEL'] = model
    config['WEIGHT_PATH'] = path
    return config

def run_test_script():
    subprocess.run(["python", "test_script.py"])

if __name__ == "__main__":
    yaml_file_path = 'your_yaml_file.yaml'  # Update with your YAML file path
    config = read_yaml(yaml_file_path)

    models = ['ResNet50', 'DenseNet121', 'DeiT-S']  # Update with your list of models
    paths = ['/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/cifar_final_hyp/resnet-50/2024-01-05_17-41-27/', \
        '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/cifar_final_hyp/densenet-121/2024-01-05_17-26-35/', \
             '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/cifar_final_hyp/deit-S/2024-01-07_18-19-48/']  # Update with your list of paths

    for model, path in zip(models, paths):
        updated_config = update_yaml(config.copy(), model, path)
        write_yaml(yaml_file_path, updated_config)
        
        # Run the test_script.py
        run_test_script()
