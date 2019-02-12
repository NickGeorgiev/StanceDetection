import pandas
import random
import json

from preprocessor import initial_text_clean_up

def parse_csv_file(csv_file_path):
    data = pandas.read_csv(csv_file_path)
    return [
        (initial_text_clean_up(tweet),target, stance)
        for (tweet, target, stance)
        in list(zip(data['Tweet'], data['Target'], data['Stance']))
    ]


if __name__ == '__main__':
    labeled_data = parse_csv_file(open('../data/stance_train.csv','r')) + parse_csv_file(open('../data/stance_test.csv','r'))
    random.shuffle(labeled_data)
    
    with open('../jsons/labeled_data.json', 'w') as file:
        file.write(json.dumps(labeled_data))

