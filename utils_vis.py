import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.pylab as pylab
from skimage import io, color, img_as_float
from skimage.exposure import adjust_gamma
import utils

def plot_loss(loss_list, xlabel, ylabel, title):
    '''
    :param loss_list: List containing total loss values
    :param recon_list: List containing reconstruction loss
    :param xlabel: string for xlabel
    :param ylabel: string for ylabel
    :param title: string for title
    :return: loss value plot
    '''
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(linestyle='dotted')
    plt.plot(loss_list)


def plot_comparison(input_img, caption, save_path=None, save_name=None, save_as='png',
                    save_dpi=300, captions_font = 20, n_row=1, n_col=2,
                    figsize=(5, 5), cmap='gray'):
    '''
    Plot comparison of multiple image but only in column wise!
    :param input_img: Input image list
    :param caption: Input caption list
    :param save_path: Path to save plot
    :param save_name: Name to be save for plot
    :param: save_as: plot save extension, 'png' by DEFAULT
    :param IMG_SIZE: Image size
    :param n_row: Number of row is 1 by DEFAULT
    :param n_col: Number of columns
    :param figsize: Figure size during plotting (5,5) by DEFAULT
    :return: Plot of (n_row, n_col)
    '''
    print()
    assert len(caption) == len(input_img), "Caption length and input image length does not match"
    assert len(input_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        axes[i].imshow(np.squeeze(input_img[i]), cmap=cmap)
        axes[i].set_xlabel(caption[i], fontsize=captions_font)
        axes[i].set_xticks([])
        axes[i].set_yticks([])

    plt.tight_layout()
    if save_path!=None:
        plt.savefig(save_path+'{}.{}'.format(save_name, save_as), save_dpi=save_dpi)
    plt.show()


def plot_hist(inp_img, titles, n_row=1, n_col=2,
              n_bin=20, ranges=[0, 1], figsize=(5, 5)):
    '''
    Plot histogram side by side
    :param inp_img: Input image stacks as list
    :param titles: Input titles as list
    :param n_row: Number of row by DEFAULT 1
    :param n_col: Number of columns by DEFAULT 2
    :param n_bin: Number of bins by DEFAULT 20
    :param ranges: Range of pixel values by DEFAULT [0,1]
    :param figsize: Figure size while plotting by DEFAULT (5,5)
    :return:
        Plot of histograms
    '''
    assert len(titles) == len(inp_img), "Caption length and input image length does not match"
    assert len(inp_img) == n_col, "Error of input images or number of columns!"

    fig, axes = plt.subplots(n_row, n_col, figsize=figsize)
    fig.subplots_adjust(hspace=0.4, wspace=0.4, right=0.7)

    for i in range(n_col):
        inp = np.squeeze(inp_img[i])
        axes[i].hist(inp.ravel(), n_bin, ranges)
        axes[i].set_title(titles[i])
        axes[i].set_xlabel('Pixel Value')
        axes[i].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# reference https://github.com/naomifridman/Unet_Brain_tumor_segmentation
def show_n_images(imgs, titles=None, enlarge=20, cmap='gray'):
    plt.set_cmap(cmap)
    n = len(imgs)
    gs1 = gridspec.GridSpec(1, n)

    fig1 = plt.figure();  # create a figure with the default size
    fig1.set_size_inches(enlarge, 2 * enlarge);

    for i in range(n):

        ax1 = fig1.add_subplot(gs1[i])

        ax1.imshow(imgs[i], interpolation='none');
        if (titles is not None):
            ax1.set_title(titles[i])
        ax1.set_xticks([])
        ax1.set_yticks([])

    plt.show();


def show_lable_on_image4(test_img, label_im):
    alpha = 0.8
    # normalizing image
    # img = img_as_float(test_img/test_img.max())
    img = utils.min_max_norm(test_img)
    rows, cols = img.shape

    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    red_multiplier = [1, 0.2, 0.2]
    green_multiplier = [0.35, 0.75, 0.25]
    blue_multiplier = [0, 0.25, 0.9]
    yellow_multiplier = [1, 1, 0.25]
    brown_miltiplier = [40. / 255, 26. / 255, 13. / 255]

    #label 1 => necrotic and non-enhancing tumor core
    color_mask[label_im == 1] = blue_multiplier  # [1, 0, 0]  # Red block
    #label 2 => peritumoral edema
    color_mask[label_im == 2] = yellow_multiplier  # [0, 1, 0] # Green block
    #label 3/4 => GD-enhancing tumor
    color_mask[label_im == 3] = brown_miltiplier  # [0, 0, 1] # Blue block
    color_mask[label_im == 4] = green_multiplier  # [0, 1, 1] # Blue block

    # Construct RGB version of grey-level image
    img_color = np.dstack((img, img, img))

    # Convert the input image and color mask to Hue Saturation Value (HSV)
    # colorspace
    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    # Replace the hue and saturation of the original image
    # with that of the color mask
    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)

    return img_masked


def plot_labels_color(label_im):
    rows, cols = label_im.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    red_multiplier = [1, 0.2, 0.2]
    green_multiplier = [0.35, 0.75, 0.25]
    blue_multiplier = [0, 0.25, 0.9]
    yellow_multiplier = [1, 1, 0.25]
    brown_multiplier = [40. / 255, 26. / 255, 13. / 255]

    color_mask[label_im == 1] = blue_multiplier  # [1, 0, 0]  # Red block
    color_mask[label_im == 2] = yellow_multiplier  # [0, 1, 0] # Green block
    color_mask[label_im == 3] = green_multiplier  # [0, 0, 1] # Blue block
    color_mask[label_im == 4] = green_multiplier  # [0, 1, 1] # Blue block

    return color_mask


def plot_labels_color255(label_im):
    rows, cols = label_im.shape
    # Construct a colour image to superimpose
    color_mask = np.zeros((rows, cols, 3))
    red_multiplier = [255, 51, 51]
    green_multiplier = [89, 191, 64]
    blue_multiplier = [0, 64, 230]
    yellow_multiplier = [255, 255, 64]
    #     brown_multiplier = [40. / 255, 26. / 255, 13. / 255]

    color_mask[label_im == 1] = blue_multiplier  # [1, 0, 0]  # Red block
    color_mask[label_im == 2] = yellow_multiplier  # [0, 1, 0] # Green block
    color_mask[label_im == 3] = green_multiplier  # [0, 0, 1] # Blue block
    color_mask[label_im == 4] = green_multiplier  # [0, 1, 1] # Blue block

    return color_mask