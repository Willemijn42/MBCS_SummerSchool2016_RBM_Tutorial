"""Create, train and test an RBM. Use Python 3.4 or up.

First create an RBM with certain numbers of hidden layer 1 (HL1) and hidden layer 2 (HL2) nodes.
Then train the network (unsupervised) using contrastive divergence (CD) learning, with method 'training'.
Training data is obtained from file 'letters3'.
Use 'letter_reproduction' to visualize learning. This method prints the original image and reproduction by the network.
'receptive_field_plot' will plot the receptive fields of some or all of the HL1 nodes.

Lastly some custom functions are included.

"""


import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import copy
import string
import math as mt
from mpl_toolkits.mplot3d import Axes3D


class RBM(object):
    """Create, train and test an RBM with one visible layer, two hidden layers and one classifier layer.

    Class methods:
    training -- unsupervised training
    letter_reproductions -- create reproductions of input data
    receptive_field_plot -- plot the receptive fields of HL1 nodes
    """

    def __init__(self, name, nvis=900, nhid1=20, nhid2=100, weights_stdev=0.01):
        """Define instance variables.

        :param name: RBM name, string, used in naming output files
        :param nvis: number of units (pixels) in visible layer (default: 900)
        :param nhid1: number of nodes in hidden layer 1 (default: 20)
        :param nhid2: number of nodes in hidden layer 2 (default: 100)
        :param weights_stdev: standard deviation in gaussian distribution of initial weights (default: 0.01)
        """

        self.name = name            # Store RBM name
        self.nvis = nvis            # Set number of visible and hidden layer units
        self.nhid1 = nhid1
        self.nhid2 = nhid2

        # Some instance variables specified in other methods, placeholder value, if possible correct type
        self.learn_rate = 0.0

        # Initialize weight and bias weight matrices:
        self.weights_stdev = weights_stdev
        self.weights1 = weights_stdev * np.random.randn(nvis, nhid1)        # to small random values for the HLs
        self.weights2 = weights_stdev * np.random.randn(nhid1, nhid2)

        self.weights1_b = np.zeros((1, nhid1))                              # to zeros for the bias weights
        self.weights2_b = np.zeros((1, nhid2))

        self.weightsvis_b = np.zeros((1, 1))                                # NOTE: will be set when data is obtained

        # Track training progress and visualize learning:
        self.completed_training_epochs = 0
        self.err_list = []
        self.perclist_w1 = []
        self.perclist_w2 = []
        self.percentile_list = [5, 25, 50, 75, 95]

    def training(self, epochs=1000, learn_rate=0.01, data_name='letters3', sety_learncurve=False,
                 file_extension=".png"):
        """Train RBM unsupervised using CD-1 learning.

        Parameters:
        :param epochs: how many times the entire data set is fed to the network (default: 1000)
        :param learn_rate: learning rate used in CD learning (default: 0.01)
        :param data_name: name of the training data (default: 'letters3') **
        :param sety_learncurve: whether to standardize y-axis range for the learning curve plot (default: False)
        :param file_extension: learning curve output file extension (default: ".png")
        :return: nothing, store learned weights in self

        ** Note: The data consists of an array, where each row corresponds to a training image.
        """

        # Load training data
        data = pickle.load(open(data_name, "rb"), encoding='latin1')
        # Record number of training images
        num_tr_im = 26

        # Calculate probability that a certain pixel in training data is on:
        p_unit_on_arr = np.mean(data, axis=0)
        # Set visible layer bias weights to log(p/(1-p))
        # TODO: divide by zero - error by following line. Why? Seems to be python 2.7 floor division problem, but my
        # solutions (convert array to float, multiply possible values of 1 by .99999) do not work. Code runs anyway
        self.weightsvis_b = np.log(p_unit_on_arr.astype(float) / (1 - (.9999 * p_unit_on_arr)))
        self.weightsvis_b = self.weightsvis_b.reshape((1, 900))
        # Record learning rate
        self.learn_rate = copy.copy(learn_rate)

        """Training Layer 1"""
        for epoch in range(epochs):
            # Shuffle data to divide randomly in mini-batches (note: variable 'data' is now scrambled)
            np.random.shuffle(data)

            """Positive phase:
            feed figure (visible layer) to the network, compute HL1 activation."""
            pos_hid1_activprob, pos_hid1_activ = act_forw(data, self.weights1, self.weights1_b)

            """Negative phase:
            calculate 'imagined' visible and HL1 activations."""
            neg_vis_activprob = act_back(pos_hid1_activ, self.weights1, self.weightsvis_b, only_activprob=True)
            # NOTE: only use activprob instead of activ, faster. Common procedure, among others reduces
            # sampling noise.
            neg_hid1_activprob = act_forw(neg_vis_activprob, self.weights1, self.weights1_b, only_activprob=True)
            # Note: no need to calculate neg_hid1_activ because it isn't used anywhere

            """Update weights using the associations between Vis and HL1 activation.
            NOTE: with 'association', the value (v_i * h_j) is meant, used in calculating
            the weight change between nodes i and j in the visible and hidden layer, resp."""
            delta = wupdate(data, pos_hid1_activprob, neg_vis_activprob, neg_hid1_activprob, learn_rate)
            # Update weight matrix:
            self.weights1 += delta

            # Same procedure for the bias weights (given bias activation is always equal to 1):
            vis_bias_delta, hl1_bias_delta = wupdate_b(data, pos_hid1_activprob, neg_vis_activprob, neg_hid1_activprob,
                                                       learn_rate)
            self.weightsvis_b += vis_bias_delta
            self.weights1_b += hl1_bias_delta

            # Check relative size of weight updates:
            relative_updw1 = delta / self.weights1
            perc_updw1 = np.percentile(relative_updw1, self.percentile_list)
            self.perclist_w1.append(perc_updw1)

            # Print current reproduction error per figure
            err_perfig = np.sum((data - neg_vis_activprob) ** 2) / float(num_tr_im)

            print('{}, HL1, epoch: {}, error: {}'.format(self.name, (self.completed_training_epochs + (epoch + 1)),
                                                         err_perfig))

            self.err_list.append(err_perfig)

        """Training Layer 2"""
        for epoch in range(epochs):
            # Shuffle data again
            np.random.shuffle(data)

            """Positive Phase:
            feed input through trained weights1 to get HL1 input and activation,
            on to compute HL2 input and activation."""
            pos_hid1_activprob, pos_hid1_activ = act_forw(data, self.weights1, self.weights1_b)
            pos_hid2_activprob, pos_hid2_activ = act_forw(pos_hid1_activ, self.weights2, self.weights2_b)

            """Negative Phase:
            calculate 'imagined' HL1 and HL2 activations."""
            neg_hid1_activprob = act_back(pos_hid2_activ, self.weights2, self.weights1_b, only_activprob=True)
            neg_hid2_activprob = act_forw(neg_hid1_activprob, self.weights2, self.weights2_b, only_activprob=True)
            # For visualizing learning (calc same error measure as during training HL1):
            neg_vis_activprob = act_back(neg_hid1_activprob, self.weights1, self.weightsvis_b, only_activprob=True)

            """Update weights using associations between HL1 and HL2 activations."""
            delta = wupdate(pos_hid1_activprob, pos_hid2_activprob, neg_hid1_activprob, neg_hid2_activprob, learn_rate)
            self.weights2 += delta

            hl1_bias_delta, hl2_bias_delta = wupdate_b(pos_hid1_activprob, pos_hid2_activprob, neg_hid1_activprob,
                                                       neg_hid2_activprob, learn_rate)
            self.weights1_b += hl1_bias_delta
            self.weights2_b += hl2_bias_delta

            # Check relative size of weight updates:
            relative_updw2 = delta / self.weights2
            perc_updw2 = np.percentile(relative_updw2, self.percentile_list)
            self.perclist_w2.append(perc_updw2)

            # Print reproduction error
            err_perfig = np.sum((data - neg_vis_activprob) ** 2) / float(num_tr_im)
            print('{}, HL2, epoch: {}, error: {}'.format(self.name, (self.completed_training_epochs + (epoch + 1)),
                                                         err_perfig))

            self.err_list.append(err_perfig)

        # Plot weight updates
        percplot(self.perclist_w1, self.percentile_list, '{}_w_upd_w1'.format(self.name))
        percplot(self.perclist_w2, self.percentile_list, '{}_w_upd_w2'.format(self.name))

        # Plot learning (= decreasing error per figure)
        fig = plt.figure()
        if sety_learncurve:
            plt.ylim(60, 150)
        plt.plot(self.err_list)
        plt.ylabel('Mean error / figure')
        plt.xlabel('Training epoch')
        plt.savefig('{}_learningcurve{}'.format(self.name, file_extension), bbox_inches='tight')
        plt.close(fig)

        # Record number of performed training epochs:
        self.completed_training_epochs += epochs

    def letter_reproduction(self, letter='w', dataname='letters3', save_im=True, disp_im=False):

        """Produce reproductions of data figures.

        Parameters:
        :param letter: desired letter to be reproduced (string) (default: 'w')
        :param dataname: name data file containing figures to be reproduced (default: 'letters3')
        :param save_im: whether to save the plot as an image file (default: True)
        :param disp_im: whether to show the image on screen (default: False)
        (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
        :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
        """

        data = pickle.load(open(dataname, 'rb'), encoding='latin1')

        """Determine which figure to reproduce"""
        alphabet = list(string.ascii_lowercase)
        letter_index = alphabet.index(letter)
        # Create array containing desired original image vector
        originals = data[letter_index, :]
        originals = originals.reshape((1, 900))             # otherwise shape = (, 900), causes broadcast error

        """Forward phase (feed activation from visible layer to HL2)."""
        for_hid1_activprob, for_hid1_activ = act_forw(originals, self.weights1, self.weights1_b)
        for_hid2_activprob, for_hid2_activ = act_forw(for_hid1_activ, self.weights2, self.weights2_b)

        """Backward Phase (calculate 'imagined' HL1 and Vis activations)."""
        back_hid1_activprob, back_hid1_activ = act_back(for_hid2_activprob, self.weights2, self.weights1_b)
        reconstructions = act_back(back_hid1_activ, self.weights1, self.weightsvis_b, only_activprob=True)

        originals = originals.reshape((30, 30))                     # Reshape for display purposes
        reconstructions = reconstructions.reshape((30, 30))

        basicplot(originals, file_name="{}_epoch_{}_{}_orig".format(self.name, self.completed_training_epochs, letter),
                  disp_im=disp_im, save_im=save_im)
        basicplot(reconstructions, file_name="{}_epoch_{}_{}_repr".format(self.name, self.completed_training_epochs,
                                                                          letter), disp_im=disp_im, save_im=save_im)

    def receptive_field_plot(self, index_list=None):
        """Print the weight distributions (receptive fields) for (a selection of) all HL1 nodes of the model.

        :param index_list: specific indices of nodes whose rec fields need to be printed (default: None (= print all))
        :return: nothing, save images of receptive fields
        """

        if not index_list:
            index_list = range(self.nhid1)
        else:
            index_list = index_list

        for index in index_list:
            file_name = "{}_recfield_hl1_node_{}".format(self.name, index)
            basiccontourplot(self.weights1[:, index], file_name)


"""Some custom functions used in RBM training and plotting."""


def sigmoid(xarr, maxval=1, theta=0, temp=1):
    """Logistic sigmoid function to transform input, used in the RBM class defined in rbm.py.

    :param xarr: elements of this object to be converted (should work for arrays and floats)
    :param maxval: upper limit (default: 1)
    :param theta: degree of freedom 1 (default: 0)
    :param temp: degree of freedom 2 (default: 1)
    :return: object of same type and shape as input (some_object)

    Note: Input is converted to a number in the range [0, 1]. A net-input of 0 returns .5.
    Approaches zero and one around -4 and 4, respectively. Output is same type and shape as input.
    """

    yarr = maxval * (1 + np.exp((- xarr - theta) / temp)) ** (-1)   # sigmoid body

    return yarr                                                     # return answer array


def act_forw(data_vis, weight_arr, hid_bias_arr, only_activprob=False):
    """Calculate HL activation probability and activation given Vis activation, weights Vis-HL, and bias weights.

    :param data_vis: data to be forwarded (or: visible layer activation)
    :param weight_arr: weight array between Vis and HL
    :param hid_bias_arr: array of HL bias weights, with same number of rows as data array
    :param only_activprob: whether only the activprob, or also activity is needed (longer calculation) (default: False)
    :return: array of hidden layer activity probability and activity, resp. Shape: (per_batch, nhid).
    Hidden layer activity will NOT be calculated or returned when only_activprob is set to True
    """

    # Compute HL input by multiplying the visible layer activation with the weight matrix:
    hid_input = np.dot(data_vis, weight_arr)
    hid_input += hid_bias_arr                    # Add bias
    # Compute HL activation probabilities using sigmoid function defined above
    hid_activprob = sigmoid(hid_input)

    if only_activprob:
        return hid_activprob
    else:
        # To get the actual activations from the probabilities, compare probability with random number [0-1]:
        hid_activ = hid_activprob > np.random.rand(hid_activprob.shape[0], hid_activprob.shape[1])
        # convert boolean activation array into integer zeros and ones:
        hid_activ = hid_activ.astype(int)

        return hid_activprob, hid_activ                 # Return result


def act_back(data_hid, weight_arr, vis_bias_arr, only_activprob=False):
    """Calculate Vis activation probability and activation given Vis activation, weights Vis-HL, and bias weights.

    :param data_hid: HL activation (/probability) vector
    :param weight_arr: weight array between Vis and HL
    :param vis_bias_arr: array of Vis bias weights with same number of rows as data array
    :param only_activprob: whether only the activprob, or also activity is needed (longer calculation) (default: False)
    :return: array of vis-layer activity probability and activity, resp. Shape: (per_batch, nvis).
    Vis_activ will NOT be calculated or returned when only_activprob is set to True
    """

    # Compute Vis input by multiplying HL activation with the transpose of the weight matrix:
    vis_input = np.dot(data_hid, np.transpose(weight_arr))
    vis_input += vis_bias_arr                   # Add bias
    # Compute Vis activation probabilities using sigmoid function defined above
    vis_activprob = sigmoid(vis_input)

    if only_activprob:
        return vis_activprob
    else:
        # To get the actual activations from the probabilities, compare probability with random number [0-1]:
        vis_activ = vis_activprob > np.random.rand(vis_activprob.shape[0], vis_activprob.shape[1])
        # Just to be sure, convert boolean activation array into integer zeros and ones:
        vis_activ = vis_activ.astype(int)
        return vis_activprob, vis_activ


def wupdate(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate):
    """Calculate update for regular weight matrices during training 1.

    :param data_vis: visible layer input data (if available, activprob rather than activ) (positive phase)
    :param pos_hid_activprob: hidden layer activprob (positive phase)
    :param neg_vis_activprob: visible layer activprob (negative phase)
    :param neg_hid_activprob: hidden layer activprob (negative phase)
    :param learn_rate: learning rate
    :return: array of weights updates. Shape: (nvis, nhid)
    """

    # Calculate the positive and negative associations:
    pos_assoc = np.dot(np.transpose(data_vis), pos_hid_activprob)
    neg_assoc = np.dot(np.transpose(neg_vis_activprob), neg_hid_activprob)

    # Compute difference between input-output (error), adjusted for batch-size:
    batch_delta = learn_rate * ((pos_assoc - neg_assoc) / data_vis.shape[0])
    # Divide by batch size to allow learning rate to be constant at different batch sizes (see Hinton)

    return batch_delta


def wupdate_b(data_vis, pos_hid_activprob, neg_vis_activprob, neg_hid_activprob, learn_rate):
    """Calculate update for bias weight matrices during training 1.

    :param data_vis: visible layer input data (if available, activprob rather than activ) (positive phase)
    :param pos_hid_activprob: hidden layer activprob (positive phase)
    :param neg_vis_activprob: visible layer activprob (negative phase)
    :param neg_hid_activprob: hidden layer activprob (negative phase)
    :param learn_rate: learning rate
    :return: two arrays of weight updates for bias weights for vis and hid layer. Shape: (1, nvis) and (1, nhid), resp.
    """

    per_batch = data_vis.shape[0]

    # Similar procedure as for normal weights, given bias activation is always equal to 1:
    pos_assoc_vis_b = np.dot(np.ones((1, per_batch)), data_vis)
    neg_assoc_vis_b = np.dot(np.ones((1, per_batch)), neg_vis_activprob)
    vis_bias_delta = learn_rate * ((pos_assoc_vis_b - neg_assoc_vis_b) / per_batch)
    # Divide by batch size to allow learning rate to be constant at different batch sizes (see Hinton)

    pos_assoc_hid_b = np.dot(np.ones((1, per_batch)), pos_hid_activprob)
    neg_assoc_hid_b = np.dot(np.ones((1, per_batch)), neg_hid_activprob)
    hid_bias_delta = learn_rate * ((pos_assoc_hid_b - neg_assoc_hid_b) / per_batch)

    return vis_bias_delta, hid_bias_delta


def basicplot(fig_array, file_name, file_extension=".png", save_im=True, disp_im=False):
    """Basic plot function for vectors or arrays (fig_array), used for plotting in fig_plot_info.

    Parameters:
    :param fig_array: array containing figures to be plotted
    :param file_name: desired output file name, string
    :param file_extension: desired output file extension (default: ".png")
    :param save_im: whether to save the plot as an image file (default: True)
    :param disp_im: whether to display the image on screen (default: False)
    (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
    :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
    """

    # Check if figure is still a vector. If so, reshape into square array.
    if len(fig_array.shape) == 1:
        side = mt.sqrt(len(fig_array))
        new_array = np.reshape(fig_array, (side, side))
    else:
        new_array = fig_array

    fig = plt.figure()

    plt.imshow(new_array, interpolation='none')         # Make image handle, remove blurring
    image_axes = plt.axes()                             # Make axes handle
    image_axes.axes.get_yaxis().set_visible(False)      # Disable y-axis
    image_axes.axes.get_xaxis().set_visible(False)      # Disable x-axis

    if disp_im:
        plt.show()                                          # To show the plot on screen in python

    if save_im:
        # Save to (.png) file, remove white border
        fig.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure


def basiccontourplot(w1_array, file_name, file_extension=".png", save_im=True, disp_im=False):
    """Contour plot function for trained RBM [VIS->HID1] weights.

    Parameters:
    :param w1_array: array of weights to be plotted
    :param file_name: desired output file name, string
    :param file_extension: desired output file extension (default: ".png")
    :param save_im: whether to save the plot as an image file (default: True)
    :param disp_im: whether to show the image on screen (default: False)
    (NOTE: showing the plot is not suitable for all circumstances! E.g. remote server, multiple sequential images)
    :return: nothing, displays or saves resulting image depending on the save_im and disp_im variables
    """

    # Check if weights are still shaped as a vector. If so, reshape into square array.
    if len(w1_array.shape) == 1:
        side = mt.sqrt(len(w1_array))
        z = np.reshape(w1_array, (side, side))
    else:
        z = w1_array

    fig = plt.figure()
    ax = Axes3D(fig)
    x, y = np.mgrid[:z.shape[0], :z.shape[1]]

    ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet)

    # ax.set_xlabel('X')
    # ax.set_xlim(-40, 40)
    # ax.set_ylabel('Y')
    # ax.set_ylim(-40, 40)
    # ax.set_zlabel('Z')
    # ax.set_zlim(-100, 100)

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        fig.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

    plt.close(fig)  # Clear current figure


def percplot(data, perclist, file_name='weight_update_percentile', file_extension='.png', save_im=True, disp_im=False,
             sety=True, ylab='Update / Weight', xlab='Epoch', xvalues=False):
    """Plot percentiles, e.g. of weight update information during 'training1' of RBM object (rbm.py)

    :param data: list of lists, each sublist containing the same percentiles of the weight update matrices
    :param perclist: list of used percentiles
    :param file_name: name of optional output file (default: 'weight_update_percentile')
    :param file_extension: extension of optional output file (default: '.png')
    :param save_im: whether to save image (default: True)
    :param disp_im: whether to show image (default: False)
    :param sety: whether to set the y-range to predefined limits (default: True)
    :param ylab: intended label for y-axis (string) (default: 'update / weight')
    :param xlab: intended label for x-axis (string) (default: 'batch number')
    :param xvalues: if desired, list / array of x values can be supplied (default: False)
    :return: nothing, displays plot or saves plot to output file
    """

    fig = plt.figure()

    if xvalues:
        lines = plt.plot(xvalues, data)
    else:
        lines = plt.plot(data)

    plt.ylabel(ylab)
    plt.xlabel(xlab)

    if sety:
        plt.ylim([-0.02, 0.02])

    plt.figlegend(lines, perclist, 'upper right', title='Percentiles')

    if disp_im:
        plt.show()

    if save_im:
        # Save to (.png) file, remove white border
        plt.savefig("{}{}".format(file_name, file_extension), bbox_inches='tight')

        plt.close(fig)  # Clear current figure
