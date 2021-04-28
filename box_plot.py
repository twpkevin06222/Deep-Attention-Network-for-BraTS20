import numpy as np
import pandas as pd
import fnmatch
import matplotlib.pyplot as plt
#%%
#file name
training_filename = 'Stats_Training_final2108wAug.csv'
validation_filename = 'Stats_Validation_final2108wAug.csv'
# specify path
training_ds_path = 'C:\\Users\\Kevin Teng\\Desktop\\BrainTumourSegmentation\\Results\\Training\\'+ training_filename
validation_ds_path = 'C:\\Users\\Kevin Teng\\Desktop\\BrainTumourSegmentation\\Results\\Validation\\'+validation_filename
# read .csv file from path
train_csv = pd.read_csv(training_ds_path)
validation_csv = pd.read_csv(validation_ds_path)


#%%
# use cases for DC
header = train_csv.columns.tolist()
train_csv = pd.read_csv(training_ds_path, header=None, index_col=None)
validation_csv = pd.read_csv(validation_ds_path, header=None, index_col=None)

n_row = train_csv.shape[0] - 5 #6 rows are redudant
#%%
def wildcard_list(query_list, query, mode=None):
    """
    Return the index list of the query list containing the wild car query
    :param query_list: List to be search
    :param query: Wild card query, use * for multiple wildcard, ? for single wildcard
    :param mode: 'index' for returning index list, 'string' for string list
    :return: List containing the query based on mode
    """
    assert mode=="index" or mode=="string", "Please input mode as 'index' or 'string'!"
    bool_list = [fnmatch.fnmatch(x, query) for x in query_list]
    idx_list = [i for i in range(len(bool_list)) if bool_list[i] is True]
    if mode == "index":
        return idx_list
    else:
        str_list = [header[j] for j in idx_list]
        return str_list

dc_col = wildcard_list(header, 'Dice*', mode="index")

# initialize list for appending data frame for each DC cases
train_dc = []
val_dc = []
# loop throught the DC cases and slice & convert to list
for dc in dc_col:
    train_dc.append(list(map(np.float32,train_csv.iloc[1:n_row, dc].tolist()))) # map from string->int
    val_dc.append(list(map(np.float32,validation_csv.iloc[1:n_row, dc].tolist()))) # map from string->int

#%%

def box_plot_plotter(data_a, data_b, figsize=(10,10),
                     title='title', xlabel='xlabel', ylabel='ylabel',
                     ticks_font = 10, label_font=15,
                     save=False, save_path=None, save_name=None, format='png', dpi=300):
    '''
    Box Plot for dual input
    @param data_a: Training Data
    @param data_b: Validation Data
    '''
    ticks = ['ET', 'WT', 'TC']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    # plt.figure(figsize)
    plt.figure()
    bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.4, sym='', widths=0.6)
    bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.4, sym='', widths=0.6)
    set_box_color(bpl, '#D7191C') # colors are from http://colorbrewer2.org/
    set_box_color(bpr, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    plt.plot([], c='#D7191C', label='Training')
    plt.plot([], c='#2C7BB6', label='Validation')
    plt.legend()
    plt.title(title, fontsize = label_font)
    plt.xlabel(xlabel, fontsize = label_font)
    plt.ylabel(ylabel, fontsize = label_font)
    plt.xticks(range(0, len(ticks) * 2, 2), ticks, fontsize= ticks_font )
    plt.xlim(-2, len(ticks)*2)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('{}_{}.{}'.format(save_path, save_name, format), format=format, dpi=dpi)

box_plot_plotter(train_dc, val_dc, figsize=(15,15), ticks_font= 10, label_font=15,
                 title='Dice Score for Deep Supervised Unet',
                 xlabel='Tumour Type', ylabel='Dice Score')

#%%
