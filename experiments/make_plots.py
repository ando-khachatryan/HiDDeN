import plotly
import plotly.graph_objs as go
import pandas as pd
import plotly.io as pio
import argparse
import os


# def make_plots(data_folder: str, output_folder: str, validation_or_train: str = 'validation'):

#     if validation_or_train == 'validation':
#         data = pd.read_csv(os.path.join(data_folder, 'validation.csv'))
#     elif validation_or_train == 'training':
#         data = pd.read_csv(os.path.join(data_folder, 'train.csv'))
#     else:
#         raise ValueError('The value of the parameter validation_or_train should be either "validation" or "training"')

#     num_epochs = max(data['epoch'])

#     metrics = [('Bitwise error', 'bitwise-error'),
#             ('Encoder Mean Square Error', 'encoder_mse'), 
#             ('Adversarial Binary Cross-Entropy', 'adversarial_bce'),
#             ]

#     for metric_name, metric_csv_column in metrics:
#         max_value = max(data[metric_csv_column])
#         min_value = min(min(data[metric_csv_column]), 0)

#         validation_scatter = go.Scatter(
#             x=data['epoch'],
#             y=data[metric_csv_column],
#             name='{}, {}'.format(metric_name, 'validation'),
#             line = dict(
#                 width = 2)
#             )
#         layout = dict(title = metric_name,
#                     xaxis=dict(title = 'Epoch', range=[0, num_epochs], zeroline=True, showline=True),
#                     yaxis=dict(title = metric_name, range = [min_value, 1.1 * max_value], zeroline=True, showline=True))

#         fig = dict(data=[validation_scatter], layout=layout)
#         filename = os.path.join(output_folder, '{}.pdf'.format(metric_name))
#         print(filename)
#         pio.write_image(fig, filename)

#     metrics = [('Cover images', 'discr_cover_bce'), ('Encoded images', 'discr_encod_bce')]

#     max_value = 0
#     min_value = 0

#     scatters = []

#     for metric_descr, metric in metrics:
#         max_value = max(max(data[metric]), max_value)
#         min_value = min(min(data[metric]), min_value)

#         scatter = go.Scatter(
#             x=data['epoch'],
#             y=data[metric],
#             name=metric_descr,
#             line = dict(width = 2)
#             )
#         scatters.append(scatter)

#     layout = dict(title = 'Discriminator Binary Cross-Entropy',
#                 xaxis=dict(title = 'Epoch', range=[0, num_epochs], zeroline=True, showline=True),
#                 yaxis=dict(title = 'BCE', range = [min_value, 1.1 * max_value], zeroline=True, showline=True))

#     fig = dict(data=scatters, layout=layout)
#     pio.write_image(fig, os.path.join(output_folder, 'Discriminator cross-entropy.pdf'))


# def main():
#     # parser = argparse.ArgumentParser(description='Tools for making plots')
#     # parser.add_argument('--data-dir', '-d', required=True, type=str)
#     # parser.add_argument('--output-dir', '-o', required=True, type=str)
#     #
#     # args = parser.parse_args()


#     # make_plots(args.data_dir, args.output_dir)
#     make_plots('./no-noise-defaults/', './')

# if __name__ == '__main__':
#     main()

# data_folder = './'
# output_folder = './'
# validation_or_train = 'validation'

extension = 'svg'

data = pd.read_csv('validation.csv')
num_epochs = max(data['epoch'])

metrics = [('Bitwise Error of the Decoder', 'Bitwise error', 'bitwise-error'),
            ('Encoder Mean Square Error', 'MSE', 'encoder_mse'),
            ('Encoder Binary Cross-Entropy', 'BCE', 'adversarial_bce'),
        ]

for plot_title, y_axis_title, metric_csv_column in metrics:
    max_value = max(data[metric_csv_column])
    min_value = min(min(data[metric_csv_column]), 0)

    validation_scatter = go.Scatter(
        x=data['epoch'],
        y=data[metric_csv_column],
        line = dict(
            width = 2)
        )
    layout = dict(title = plot_title,
                xaxis=dict(title = 'Epoch', range=[0, num_epochs], zeroline=True, showline=True),
                yaxis=dict(title = y_axis_title, range = [min_value, 1.1 * max_value], zeroline=True, showline=True))

    fig = dict(data=[validation_scatter], layout=layout)
    filename = '{}.{}'.format(metric_csv_column, extension)
    print(filename)
    pio.write_image(fig, filename)

metrics = [('Cover images', 'discr_cover_bce'), ('Encoded images', 'discr_encod_bce')]

max_value = 0
min_value = 0

scatters = []

for metric_descr, metric in metrics:
    max_value = max(max(data[metric]), max_value)
    min_value = min(min(data[metric]), min_value)

    scatter = go.Scatter(
        x=data['epoch'],
        y=data[metric],
        name=metric_descr,
        line = dict(width = 2)
        )
    scatters.append(scatter)

layout = dict(title = 'Discriminator Binary Cross-Entropy',
            xaxis=dict(title = 'Epoch', range=[0, num_epochs], zeroline=True, showline=True),
            yaxis=dict(title = 'BCE', range = [min_value, 1.1 * max_value], zeroline=True, showline=True))

fig = dict(data=scatters, layout=layout)
filename = 'discriminator_bce.{}'.format(extension)
print(filename)
pio.write_image(fig, filename)





