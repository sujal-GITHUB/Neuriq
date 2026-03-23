import os
import re

def replace_in_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        return
    
    new_content = content
    
    # Direct string replacements (case sensitive) to avoid word boundary issues
    replacements = {
        'NeuroAnxiety': 'Neuriq',
        'anxiety_level': 'anxiety_level',
        'ANXIETY_CLASSES': 'ANXIETY_CLASSES',
        'AnxietyMeter': 'AnxietyMeter',
        'anxietyCards': 'anxietyCards',
        'getAnxietyColor': 'getAnxietyColor',
        'getAnxietyBgClass': 'getAnxietyBgClass',
        'Anxiety Classification Levels': 'Anxiety Classification Levels',
        'Anxiety levels': 'Anxiety levels',
        'Anxiety Levels': 'Anxiety Levels',
        'Anxiety Detected': 'Anxiety Detected',
        'anxiety state': 'anxiety state',
        'Anxiety marker': 'Anxiety marker',
        'anxiety indicator': 'anxiety indicator',
        'anxiety': 'anxiety',
        'Anxiety': 'Anxiety',
        'ANXIETY': 'ANXIETY'
    }
    
    # We apply them in order, with the most specific first, but wait:
    # 'anxiety' -> 'anxiety' will just replace all substrings, which is what we want!
    # So we just do a simple case-sensitive substring replace for the core words.
    
    # Wait, 'NeuroAnxiety' was already replaced in the first script? Maybe partially.
    # Let's just do a blanket replace for 'anxiety', 'Anxiety', 'ANXIETY' 
    new_content = new_content.replace('Anxiety', 'Anxiety')
    new_content = new_content.replace('anxiety', 'anxiety')
    new_content = new_content.replace('ANXIETY', 'ANXIETY')

    if new_content != content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"Updated {filepath}")

def walk_dir(directory):
    for root, dirs, files in os.walk(directory):
        if 'node_modules' in root or '.git' in root or '.next' in root or 'venv' in root:
            continue
        for file in files:
            if file.endswith(('.ts', '.tsx', '.py', '.txt', '.md', '.json', '.css', '.js')):
                filepath = os.path.join(root, file)
                replace_in_file(filepath)
                
                # Also rename file if needed
                new_file = file.replace('Anxiety', 'Anxiety').replace('anxiety', 'anxiety')
                if new_file != file:
                    new_filepath = os.path.join(root, new_file)
                    os.rename(filepath, new_filepath)
                    print(f"Renamed {filepath} to {new_filepath}")

if __name__ == '__main__':
    walk_dir('d:/Work/Projects/Neuriq')
