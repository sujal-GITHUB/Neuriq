import os
import re

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return
    
    # Track if we made changes
    new_content = content
    
    # Specific variables and names
    replacements = {
        'NeuroAnxiety': 'Neuriq', # Actually it's Neuriq maybe? User says "anxiety detection" and "XGBoost + LightGBM + CatBoost Ensemble". Let's name it NeuroAnxiety or just keep it NeuroAnxiety/NeuroAnxiety. Let's use NeuroAnxiety.
        'NeuroAnxiety': 'NeuroAnxiety',
        'anxiety_level': 'anxiety_level',
        'ANXIETY_CLASSES': 'ANXIETY_CLASSES',
        'AnxietyMeter': 'AnxietyMeter',
        'anxietyCards': 'anxietyCards',
        'getAnxietyColor': 'getAnxietyColor',
        'getAnxietyBgClass': 'getAnxietyBgClass',
        'anxiety-': 'anxiety-',
        'Anxiety Classification Levels': 'Anxiety Classification Levels',
        'Anxiety levels': 'Anxiety levels',
        'Anxiety Levels': 'Anxiety Levels',
        'Anxiety Detected': 'Anxiety Detected',
        'anxiety state': 'anxiety state',
        'Anxiety marker': 'Anxiety marker',
        'anxiety indicator': 'anxiety indicator'
    }
    
    # Generic case-sensitive replacements
    new_content = re.sub(r'\bAnxiety\b', 'Anxiety', new_content)
    new_content = re.sub(r'\banxiety\b', 'anxiety', new_content)
    new_content = re.sub(r'\bANXIETY\b', 'ANXIETY', new_content)

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")
        
        # If it's a file that needs renaming
        new_path = filepath.replace('Anxiety', 'Anxiety').replace('anxiety', 'anxiety')
        if new_path != filepath:
            os.rename(filepath, new_path)
            print(f"Renamed to {new_path}")

def walk_dir(directory):
    for root, dirs, files in os.walk(directory):
        if 'node_modules' in root or '.git' in root or '.next' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith(('.ts', '.tsx', '.py', '.txt', '.md', '.json', '.css', '.js')):
                replace_in_file(os.path.join(root, file))

if __name__ == '__main__':
    walk_dir('d:/Work/Projects/Neuriq')
