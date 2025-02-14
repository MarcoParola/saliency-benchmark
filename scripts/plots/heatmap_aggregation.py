import os


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_heatmap(data, title,i):
    plt.figure(figsize=(10, 6))
    ax = sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    #ax.set_title(title, pad=20)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top')  # Move xlabel to the top
    ax.set_xlabel("Extractor-Concept presence method")  # Add label above heatmap
    if i==3 or i==4:
        ax.set_ylabel("Classes")
    else:
        ax.set_ylabel("Model-Saliency method")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    output_dir = os.path.abspath('confusion_matrices')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.tight_layout()
    plt.savefig(os.path.join("confusion_matrices", "heatmap" + str(i) + ".jpg"))
    plt.close()


# def plot_heatmap(data, title, i):
#     plt.figure(figsize=(10, 6))
#     ax = sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#     ax.xaxis.tick_top()
#     ax.set_xlabel("")  # Remove the "None-None" label
#
#     # Set original x-tick labels (for the columns)
#     ax.set_xticklabels(data.columns, rotation=0)
#
#     # Add custom labels above the first half and second half of the columns
#     # Get the number of columns in the data
#     num_cols = len(data.columns)
#
#     # For the first half of the columns (first 3)
#     for col in range(num_cols // 2):
#         ax.text(col, -0.5, "Classes_first3", ha="center", va="center", fontsize=12)
#
#     # For the second half of the columns (last 3)
#     for col in range(num_cols // 2, num_cols):
#         ax.text(col, -0.5, "Classes_last3", ha="center", va="center", fontsize=12)
#
#     if i == 3 or i == 4:
#         ax.set_ylabel("Classes")
#
#     plt.xticks(rotation=0)
#     plt.yticks(rotation=0)
#
#     output_dir = os.path.abspath('confusion_matrices')
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     plt.savefig(os.path.join("confusion_matrices", "heatmap" + str(i) + ".jpg"))
#     plt.close()

# Table 1: Imagenette Results
data1 = {
    ('GDino', 'CASP'):  [8.46, 7.88, 9.18, 8.24, 8.94, 8.32, 7.53, 7.66],
    ('GDino', 'IOU'):   [7.67, 5.76, 6.75, 6.97, 5.69, 5.71, 6.57, 7.79],
    ('GDino', 'CAS'):   [7.14, 5.92, 7.36, 7.19, 5.92, 5.96, 7.13, 6.48],
    ('Florence', 'CASP'): [9.25, 6.97, 8.13, 9.20, 8.53, 8.14, 7.39, 8.79],
    ('Florence', 'IOU'):  [9.39, 8.72, 8.81, 9.70, 8.78, 8.85, 10.52, 9.79],
    ('Florence', 'CAS'):  [7.07, 6.73, 8.28, 7.37, 6.67, 7.17, 7.66, 7.47]
}
index1 = [('ResNet', 'SIDU'), ('ResNet', 'RISE'), ('ResNet', 'LIME'), ('ResNet', 'GradCAM'),
          ('VGG', 'SIDU'), ('VGG', 'RISE'), ('VGG', 'LIME'), ('VGG', 'GradCAM')]
df1 = pd.DataFrame(data1, index=pd.MultiIndex.from_tuples(index1, names=['Model', 'Method']))
plot_heatmap(df1, "Imagenette Results",1)

# Table 2: Intel Image Results
data2 = {
    ('GDino', 'CASP'):  [9.52, 11.29, 13.39, 11.42, 12.55, 11.13, 11.90, 11.18],
    ('GDino', 'IOU'):   [14.02, 16.05, 12.93, 14.74, 17.31, 16.26, 12.96, 13.25],
    ('GDino', 'CAS'):   [10.46, 10.23, 11.71, 11.48, 10.30, 10.29, 13.46, 10.41],
    ('Florence', 'CASP'): [11.66, 6.16, 7.31, 8.55, 7.80, 7.63, 5.60, 8.25],
    ('Florence', 'IOU'):  [13.69, 13.14, 13.73, 14.16, 13.32, 13.34, 14.98, 13.52],
    ('Florence', 'CAS'):  [7.18, 6.45, 7.69, 7.00, 6.71, 6.54, 5.78, 6.63]
}
df2 = pd.DataFrame(data2, index=pd.MultiIndex.from_tuples(index1, names=['Model', 'Method']))
plot_heatmap(df2, "Intel Image Results",2)

# Table 3: Imagenette Class Mean Results
data3 = {
    ('GDino', 'CASP'):  [22.41, 15.50, 13.01, 5.71, 6.33, 2.01, 1.56, 7.16, 13.17, -4.12],
    ('GDino', 'IOU'):   [23.69, 17.88, 13.49, 4.85, 3.04, 3.03, 2.87, 2.20, -0.13, -4.79],
    ('GDino', 'CAS'):   [19.25, 13.29, 12.86, 4.82, 3.35, 2.07, 1.54, 5.55, 10.07, -6.44],
    ('Florence', 'CASP'): [16.53, 14.42, 10.44, 6.90, 7.67, 1.04, 1.73, 14.89, 8.24, 1.15],
    ('Florence', 'IOU'):  [22.85, 9.14, 15.70, 19.39, 3.88, 1.21, 3.28, 9.53, 2.11, 6.12],
    ('Florence', 'CAS'):  [14.05, 11.70, 8.96, 4.53, 5.12, 0.23, 2.05, 17.50, 7.61, 1.26]
}
index3 = ['Church', 'English Springer', 'Cassette Player', 'Golf Ball', 'Gas Pump',
          'Chain Saw', 'French Horn', 'Parachute', 'Garbage Truck', 'Tench']
df3 = pd.DataFrame(data3, index=index3)
plot_heatmap(df3, "Imagenette Class Mean Results",3)

# Table 4: Intel Image Class Mean Results
data4 = {
    ('GDino', 'CASP'):  [17.27, 15.99, 7.92, 12.29, 11.60, 5.05],
    ('GDino', 'IOU'):   [23.56, 16.79, 14.97, 13.47, 12.73, 8.16],
    ('GDino', 'CAS'):   [18.93, 9.90, 10.18, 13.92, 8.62, 5.16],
    ('Florence', 'CASP'): [9.60, 4.36, 12.64, 5.19, 9.30, 2.81],
    ('Florence', 'IOU'):  [17.49, 12.14, 16.17, 15.03, 11.29, 10.62],
    ('Florence', 'CAS'):  [9.77, 4.14, 9.33, 5.51, 6.96, 2.78]
}
index4 = ['Street', 'Forest', 'Glacier', 'Buildings', 'Sea', 'Mountain']
df4 = pd.DataFrame(data4, index=index4)
plot_heatmap(df4, "Intel Image Class Mean Results",4)


# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from matplotlib.table import Table
#
# def save_heatmap(data, title, filename):
#     plt.figure(figsize=(10, 6))
#     ax = sns.heatmap(data, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
#     ax.set_title(title, pad=20)
#     ax.xaxis.tick_top()
#     plt.xticks(rotation=45, ha='left')
#     plt.yticks(rotation=0)
#     plt.savefig(os.path.join('confusion_matrices',f"table_heatmap.jpg"), bbox_inches='tight')  # Save heatmap as an image
#     plt.close()
#
# def create_table_with_heatmap(data, title, heatmap_filename):
#     fig, ax = plt.subplots(figsize=(10, 8))
#
#     # Hide axes
#     ax.xaxis.set_visible(False)
#     ax.yaxis.set_visible(False)
#     ax.set_frame_on(False)
#
#     # Create table structure
#     table = Table(ax, bbox=[0, 0, 1, 1])
#
#     nrows, ncols = data.shape
#     width, height = 1.0 / (ncols + 1), 1.0 / (nrows + 1)  # Cell dimensions
#
#     # Add column headers
#     for j, column in enumerate(data.columns):
#         table.add_cell(0, j + 1, width, height, text=str(column), loc='center', facecolor='lightgrey')
#
#     # Add row headers
#     for i, index in enumerate(data.index):
#         table.add_cell(i + 1, 0, width, height, text=str(index), loc='center', facecolor='lightgrey')
#
#     # Add data cells
#     for i, (index, row) in enumerate(data.iterrows()):
#         for j, cell_value in enumerate(row):
#             table.add_cell(i + 1, j + 1, width, height, text=f"{cell_value:.2f}", loc='center')
#
#     ax.add_table(table)
#
#     # Insert the heatmap image
#     heatmap_img = plt.imread(heatmap_filename)
#     ax.imshow(heatmap_img, aspect='auto', extent=[1.2, 3.5, 0, nrows + 1])
#
#     plt.title(title)
#     plt.show()
#
# # Example: Using Imagenette Results
# output_dir = os.path.abspath('confusion_matrices')
# if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
# heatmap_file = os.path.join("confusion_matrices", "heatmap1.jpg")
# save_heatmap(df1, "Imagenette Results", heatmap_file)
# create_table_with_heatmap(df1, "Imagenette Results", heatmap_file)


