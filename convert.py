# convert the data from the list_path given in argument to a DyToX style dataset, and store it in the output directory
import os
import sys
import pandas as pd

if len(sys.argv) != 3:
    print('Arguments: liste_path output_dir')
    sys.exit(-1)

list_path = sys.argv[1]
output_dir = sys.argv[2]


for data_type in ['train', 'test']:
    images_list_file = os.path.join(list_path, data_type + '.lst')

    df = pd.read_csv(images_list_file, sep=' ', names=['paths','class'])
    root_folder = df['paths'].head(1)[0]
    df = df.tail(df.shape[0] -1)
    df.drop_duplicates()
    df['paths'] = df['paths'].apply(lambda x: os.path.join(root_folder, x))
    true_type = 'train/' if data_type == 'train' else 'val/'
    df['class'] = df['class'].apply(lambda x: true_type+'n'+format(int(x), '05d'))

    df = df.sort_values('class')

    # transform the dataframe to a list of tuples (path, class) and store it in a list
    if data_type == 'train':
        train_list = list(zip(df['paths'].tolist(), df['class'].tolist()))
    else:
        test_list = list(zip(df['paths'].tolist(), df['class'].tolist()))

total_list = train_list + test_list

# parallelize the copy of the images in the output directory
import multiprocessing as mp
import shutil

for elt in total_list:
    os.makedirs(os.path.join(output_dir, elt[1]), exist_ok=True)

def copy_image(image):
    shutil.copy(image[0], os.path.join(output_dir, image[1]))

with mp.Pool() as pool:
    pool.map(copy_image, total_list)

