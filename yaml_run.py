import yaml
import subprocess

def read_yaml(file_path):
    config = yaml.load(open(file_path, 'r'), Loader=yaml.FullLoader)
    return config

def write_yaml(file_path, data):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def update_yaml(base_config, model, path, val_sort_by, test_sort_by):
    config = base_config.copy()
    config['MODEL'] = model
    config['WEIGHT_PATH'] = path
    config['val_sort_by'] = val_sort_by
    config['test_sort_by'] = test_sort_by
    return config

def run_test_script():
    subprocess.run(["python", "test_script.py"])

if __name__ == "__main__":
    yaml_file_path = './configs/test_config.yaml'  # Update with your YAML file path
    base_config = read_yaml(yaml_file_path)

    models = ['ResNet50', 'DenseNet121', 'DeiT-S']  # Update with your list of models
    paths = ['/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/aptos_final_hyp/resnet50_imagenet/2024-01-10_03-03-01/',
             '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/aptos_final_hyp/densenet121_imagenet/2024-01-13_00-45-29/',
             '/l/users/santosh.sanjeev/model_soups/runs/san_final_hyp_models/san-finetune/aptos_final_hyp/deitS_imagenet/2024-01-10_02-53-56/']  # Update with your list of paths
    val_sort_by = 'Val F1'
    test_sort_by = 'Test F1'

    for model, path in zip(models, paths):
        updated_config = update_yaml(base_config, model, path, val_sort_by, test_sort_by)
        write_yaml(yaml_file_path, updated_config)

        # Run the test_script.py
        run_test_script()
