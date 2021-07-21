import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from urllib.parse import urlparse

parser = argparse.ArgumentParser(description="Analyze annotator quality.")

parser.add_argument('-a','--annotator_data_dir', type=str, help='Provide annotator data dir')
parser.add_argument('-r', '--reference_data_dir', type=str, help='Provide reference data dir')
parser.add_argument('-o', '--operation', type=str, help='What operation?  \n Operations are [count_annotator, '
                                                        'annotation_time, annotator_work, find_conflict, '
                                                        'extra_output, validate_reference_data, annotator_accuracy ]')

args = parser.parse_args()


def get_data(path, *argv):
    """ create pandas dataframe.

    :param path:
    :param argv:
    :return:
    """
    data = pd.read_json(path)
    result_data = data['results']['root_node']['results']
    requested_data = []
    requested_data_dataframe = []
    for arg in argv:
        for keys in result_data:
            for i in result_data[keys]['results']:
                requested_data.append(i[arg])
        requested_data_dataframe.append(pd.DataFrame(requested_data))
        requested_data = []
    return pd.concat(requested_data_dataframe, axis=1, join='inner')


def count_annotator(path):
    """ Calculate total annotator.
    The total annotator participated in annotation process are calculated.

    :param path: Path of annotator data file
    """
    data_frame_of_annotator = get_data(path, 'user')
    total_annotator = len(data_frame_of_annotator['vendor_user_id'].unique())
    print(f'total annotator = {total_annotator}')


def plot_annotator_work(path):
    """

    :param path: Path of annotator data file
    """
    data_frame_of_annotator = get_data(path, 'task_output', 'user', 'root_input')
    annotator_column_df = data_frame_of_annotator.pivot_table(values='answer', index=data_frame_of_annotator.index,
                                                              columns='vendor_user_id', aggfunc='first')
    annotator_work = annotator_column_df.notnull().sum()

    ax = annotator_work.sort_values(ascending=True).plot(kind='barh', width=0.85, figsize=(10, 10))
    ax.set_title('Images annotated by Annonator')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for i, v in enumerate(annotator_work.sort_values(ascending=True)):
        ax.text(v + 3, i - 0.15, str(v))
    #fig = ax.get_figure()
    #fig.tight_layout()
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    plt.show()


def plot_annotator_time(path):
    """

    :param path: Path of annotator data file
    """
    data_frame_of_annotator = get_data(path, 'task_output', 'user')
    annotator_column_df = data_frame_of_annotator.pivot_table(values='duration_ms', index=data_frame_of_annotator.index,
                                                              columns='vendor_user_id', aggfunc='first')
    print([annotator_column_df.describe()[i] for i in annotator_column_df.describe()])
    for i in annotator_column_df:
        condition = annotator_column_df[i] < 0
        if condition.any():
            value = annotator_column_df[i][annotator_column_df[i] > 0].mean()
            annotator_column_df[i].where(annotator_column_df[i] >= 0, value, inplace=True)
    ax=annotator_column_df.plot(kind='box', figsize=(10,10), vert=False)
    ax.set_title('Box plot of time taken by annotator\'s')
    ax.set_xlabel('Time in ms')
    plt.show()


def get_file_name(url):
    return os.path.basename(urlparse(url).path)[:-4]


def insert_image_name(path):
    data_frame_of_annotator = get_data(path, 'task_output', 'user', 'root_input')
    image_name_df = data_frame_of_annotator['image_url'].apply(get_file_name).rename('images')
    return pd.concat([image_name_df, data_frame_of_annotator], axis=1, join='inner')


def find_conflict_images(path):
    """

    :param path: Path of annotator data file
    """
    dataframe_with_image_name = insert_image_name(path)
    dataframe_with_image_name['answer'][dataframe_with_image_name['answer'] == 'no'] = 0
    dataframe_with_image_name['answer'][dataframe_with_image_name['answer'] == 'yes'] = 1
    img_and_ans_dataframe = dataframe_with_image_name[['answer', 'images']]
    img_and_ans_dataframe.fillna(0)
    image_data_series = img_and_ans_dataframe.groupby(['images'])
    conflict_images_list = []
    for i in image_data_series:
        df = i[1]
        df['answer'].replace('', np.nan, inplace=True)
        df.dropna(subset=['answer'], inplace=True)
        total_yes = df['answer'].sum()
        if total_yes == len(df['answer']):
            total_no = 0
        else:
            total_no = abs(total_yes - len(df['answer']))
        if abs(total_no - total_yes) <= 2:
            conflict_images_list.append(i[0])
    print(f'Annotator are disagree on these images= {conflict_images_list}')


def plot_output_detail(path):
    """

    :param path: Path of annotator data file
    """
    data_frame_of_annotator = get_data(path, 'task_output', 'user', 'root_input')
    data_f = data_frame_of_annotator[['cant_solve', 'corrupt_data', 'vendor_user_id']]
    c_data = data_f[(data_f.cant_solve == True) | (data_f.corrupt_data == True)]
    output_solve = c_data.pivot_table(values='cant_solve', index=c_data.index, columns='vendor_user_id', aggfunc='first')
    output_corrupt = c_data.pivot_table(values='corrupt_data', index=c_data.index, columns='vendor_user_id', aggfunc='first')
    width = 0.35
    ind = np.arange(output_solve.shape[1])
    fig, ax = plt.subplots(figsize=(10,10))
    rects1 = ax.bar(ind - width / 2, output_solve.sum(), width, label='Can not solve count')
    rects2 = ax.bar(ind + width / 2, output_corrupt.sum(), width,label='Corrupt data count')

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)      
    ax.set_ylabel('Count')
    ax.set_title('Annotator response for can not solve and corrupt data')
    ax.set_xticks(ind)
    ax.set_xticklabels([i for i in output_solve])
    plt.xticks(rotation=70)
    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
    fig.tight_layout()
    ax.legend()
    plt.show()


def validate_reference_data(reference_file_path):
    """

    :param reference_file_path: Path of reference data file
    """
    if reference_file_path:
        reference_data = pd.read_json(reference_file_path)
        plotting_data = [reference_data.sum(axis=1)['is_bicycle'],
                         reference_data.shape[1] - reference_data.sum(axis=1)['is_bicycle']]
        labels = ['Images with bicycle', 'Images without bicycle']
        fig, ax = plt.subplots(figsize=(10,10))
        ax.set_title('Reference data percentages')
        ax.pie(plotting_data, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)
        ax.axis('equal')
        fig.tight_layout()
        
        plt.show()
    else:
        print(f'Please prove reference data file path. \n \
        It is done using > ´python solution_1.py  anonymized_project.json -o PATH´')


def plot_annotator_accuracy(path, reference_file_path):
    """

    :param path: Path of annotator data file
    :param reference_file_path: Path of reference data file
    """
    reference_data = pd.read_json(reference_file_path)
    reference_data_column = reference_data.T.unstack().reset_index(level=1, name='r_answer').rename( \
        columns={'level_1': 'images'})[['r_answer', 'images']]
    annotators = []
    evaluation = []
    dataframe_with_image_name = insert_image_name(path)
    annotator_dataframe_series = dataframe_with_image_name.groupby('vendor_user_id')
    for i in annotator_dataframe_series:
        df = reference_data_column.merge(i[1][['answer', 'images']], left_on='images', right_on='images', how='inner')
        num_of_correct_answers = df[(df['answer'] == 'yes') & (df['r_answer'] == True)].shape[0] + \
                                 df[(df['answer'] == 'no') & (df['r_answer'] == False)].shape[0]
        percentage_of_correct_answers = (num_of_correct_answers / df.shape[0]) * 100
        evaluation.append(percentage_of_correct_answers)
        annotators.append(i[0])
    average_accuracy= sum(evaluation)/len(evaluation)
    print(average_accuracy)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.barh(annotators, evaluation)
    ax.set_title('Annonator accuracy')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    for i, v in enumerate(evaluation):
        ax.text(v + 3, i - 0.15,"{:.2f}".format(round(v, 2))+'%')
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    fig.tight_layout()
    plt.show()


def main():
    if args.operation == 'count_annotator':
        count_annotator(args.annotator_data_dir)
    elif args.operation == 'annotation_time':
        plot_annotator_time(args.annotator_data_dir)
    elif args.operation == 'annotator_work':
        plot_annotator_work(args.annotator_data_dir)
    elif args.operation == 'find_conflict':
        find_conflict_images(args.annotator_data_dir)
    elif args.operation == 'extra_output':
        plot_output_detail(args.annotator_data_dir)
    elif args.operation == 'validate_reference_data':
        validate_reference_data(args.reference_data_dir)
    else:
        plot_annotator_accuracy(args.annotator_data_dir, args.reference_data_dir)


if __name__ == "__main__":
    main()
