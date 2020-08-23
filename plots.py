
import matplotlib.pyplot as plt
import matplotlib
import sentence_grounding_test_parameters as param
import numpy as np

def plot_hidden_state_activation(sentence,
                                 activations,
                                 state = 'cell',
                                 units_to_plot = 'all',
                                 add_to_legend = None,
                                 plot_variation = False,
                                 plot_sum = False):
    """
        Plot the evolution of inner state of a RNN during the processing of a sentence.

        Arguments:
            sentence - the sentence of the analysis.
            activations - a list of numpy array of dimension (sentence length, state dimension)
                        with length the nb of state types.

            state - wich state of the activation to plot. It is intended to be used with a
                    LSTM or an ESN from the code in the files CSL_ESN.py and CSL_LSTM.py
                    The possible values are 'cell', 'state', 'output' for a LSTM and
                    'reservoir' for an ESN.
            units_to_plot - If not 'all', a list of indexes from the state dimension to plot.
            add_to_legend - A list of sring with length the state dimension to add to graph
                            legend.
            plot_variation - If True, the absolute variation in activation will be plotted.
                            Else, the value of the activation will be plotted.
            plot_sum - If True, the sum of the units will be considered, else each units will
                    be plotted with an idependant line.
    """

    words = sentence.split(" ")

    if state == 'cell':
        nb = 2
    elif state == 'state':
        nb = 1
    elif state == 'output':
        nb=0
    elif state == 'reservoir':
        nb = 0
    else:
        print("Not a valid state name")
        return

    fig = plt.figure()

    if units_to_plot == 'all':
        units_to_plot = np.arange(activations[nb].shape[1])

    if add_to_legend is None:
        add_to_legend = ["" for u in range(activations[nb].shape[1])]

    cmap = matplotlib.cm.get_cmap('tab20')


    vector_sum = np.zeros(activations[nb][:, 0].shape)

    for u in units_to_plot:
        if not(plot_sum):
            if plot_variation:
                variations = np.concatenate([[0.], np.abs( np.diff(activations[nb][:, u]))])
                plt.plot(variations, label=state+" unit "+str(u) + " "+add_to_legend[u], color = cmap(u%20))
            else:
                plt.plot(activations[nb][:, u], label=state+" unit "+str(u) + " "+add_to_legend[u], color = cmap(u%20))
        else:
            if plot_variation:
                vector_sum += np.concatenate([[0],np.abs( np.diff(activations[nb][:, u]))])
            else:
                vector_sum += activations[nb][:, u]

    if plot_sum:
        plt.plot(vector_sum, label=state+" units activation summed ("+str(units_to_plot) + ") "+add_to_legend[u])



    ax = fig.axes[0]

    ax.set_xticks(np.arange(len(words)))
    ax.set_xticklabels(words, fontsize = 10)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = 10)

    if plot_variation:
        plt.title("Variation of the activation of "+state+" neurons during the hearing of the sentence")
    else:
        plt.title("Activation of "+state+" neurons during the hearing of the sentence")

    fig.legend()

    fig.show()





def plot_concept_activation(sentence, activations, concepts_delim,nb_concepts, output_function = None, savefig=False, sub_ttl_fig='', ylims=(-0.4, 1.4)):
    """
        Plots activation through time of the different concepts while hearing
        the sentence. If output_function is not None, it is applied to the reservoir
        output vector before plotting.
        This is a reuse of a function developped by Alexis Juven.
    """

    outputs = activations.copy()
    activation_threshold = 0.5

    if output_function is not None:

        activation_threshold = output_function(activation_threshold)

        for i in range(outputs.shape[0]):
            outputs[i, :] = output_function(outputs[i, :])


    words = sentence.split(" ")
    max_nb_seen_objects = 2
    nb_object_properties = 3

    fig, axes = plt.subplots(nb_object_properties, max_nb_seen_objects, figsize=(25,20))


    concept_delimitations = [t[0] for t in concepts_delim] + [concepts_delim[-1][1]]

    for i in range(max_nb_seen_objects):

        offset = i * nb_concepts

        axes[0, i].set_title("Object " + str(i+1), fontsize = 22)

        for j in range(nb_object_properties):

            ax = axes[j, i]
            ax.plot(outputs[:, offset + concept_delimitations[j] : offset + concept_delimitations[j+1]], linewidth = 4)
            ax.legend(param.CONCEPT_LISTS[j], loc = 2, fontsize = 22)

            ax.set_yticks([0., 0.5, 1.])
            ax.set_yticklabels([0., 0.5, 1.], fontsize = 20)

            ax.set_ylim([ylims[0], ylims[1]])

            ax.set_xticks(np.arange(len(words)))
            ax.set_xticklabels(words, fontsize = 24)
            # Rotate the tick labels and set their alignment.
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize = 22)

            ax.plot(len(words) * [activation_threshold], '--', color = 'grey', linewidth = 3)

    fig.suptitle(sentence, fontsize = 26)
    plt.subplots_adjust(hspace = 0.3)

    if savefig:
        plt.savefig('sentence_'+sub_ttl_fig+".png", bbox_inches='tight')
        plt.close()
    else:
        fig.show()
        #plt.show()



def plot_final_activation(output_vect, concepts_delimitations, output_id_to_concept_dict, nb_concepts, sentence = ""):
    """
        Plot the final activation contained in output_vect with bar graph.
        This representation is useful when the number of dimension in
        the output_vect is big.
    """

    x_coo = 0
    concept_cat = 0
    spacing = 0.5
    spacing_cat = 0.7
    spacing_obj = 1.5
    Xs = []
    labels = []
    colors = []
    nb_objects = 2

    for obj in range(nb_objects):
        for i in range(obj*nb_concepts, (obj+1)*nb_concepts):

            if concepts_delimitations[concept_cat][1] == i-obj*nb_concepts:
                x_coo += spacing_cat
                concept_cat = (concept_cat +1)%len(concepts_delimitations)
            x_coo += spacing
            Xs.append(x_coo)
            labels.append("object "+str(obj+1)+" "+ output_id_to_concept_dict[i-obj*nb_concepts])

            if concept_cat == 2: #show the corresponding color
                colors.append(output_id_to_concept_dict[i-obj*nb_concepts][1:-5])
            else:
                col = [0.1,0.1,0.1, 1] #esle show a dark color different for each obejct
                col[obj] = 0.3
                colors.append(tuple(col))
        x_coo += spacing_obj
        concept_cat = 0

    Xs = [Xs[-1] - x for x in Xs] #make object 1 at the top

    fig = plt.figure()
    plt.title("Final activation\n"+sentence)
    plt.barh(Xs, list(output_vect),height = 0.4, tick_label = labels, color = colors)

    font = { #font control
            'weight' : 'normal',
            'size'   : 8}

    plt.tight_layout()
    plt.rc('font', **font)
    plt.show()



def plot_output_values_matrix(output_vect, concepts_delim, output_id_to_concept_dict, nb_concepts, sentence = "", savefig=False, sub_ttl_fig=''):
    """
        Plots a matrix showing which concepts are activated after hearing the sentence.
        Useful when the number of dimension in the output_vect is small.
        This is a reuse of a function developped by Alexis Juven.
    """
    max_nb_seen_objects = 2
    values_matrix = output_vect.T.reshape(max_nb_seen_objects, -1)

    cropped_values_matrix = np.clip(values_matrix, 0., 1.)

    fig, ax = plt.subplots(figsize=(15,10))

    ax.imshow(cropped_values_matrix, cmap = plt.get_cmap('Greys'), vmin=0., vmax=1.)

    ax.set_xticks(np.arange(nb_concepts))
    ax.set_xticklabels(list(output_id_to_concept_dict.values()))

    concept_delimitations = [t[0] for t in concepts_delim] + [concepts_delim[-1][1]]



    ax.set_yticks(range(max_nb_seen_objects))
    ax.set_yticklabels(['object ' + str(i+1) for i in range(max_nb_seen_objects)], fontsize = 16)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor", fontsize = 16)

    # Loop over data dimensions and create text annotations.
    nb_object_properties = 3
    for i in range(max_nb_seen_objects):

        for j in range(nb_object_properties):

            offset = i*nb_concepts+concept_delimitations[j]

            id_to_plot_in_red = np.argmax(output_vect[offset:i*nb_concepts+concept_delimitations[j+1]])


            for k in range(len(param.CONCEPT_LISTS[j])):

                matrix_column_id = concept_delimitations[j] + k

                value = values_matrix[i, matrix_column_id]
                value_to_plot = int(100. * value)/100.
                font = { #font control
                        'weight' : 'normal',
                        'size'   : 10}

                if k == id_to_plot_in_red:
                    color = 'r'
                    font['weight'] = 'bold'
                else:
                    color = 'black'

                ax.text(matrix_column_id, i, value_to_plot, ha="center", va="center",
                        color= color, fontsize=12)

    ax.set_title(sentence, fontsize = 22)
    fig.tight_layout()
    if savefig:
        plt.savefig('matrix_'+sub_ttl_fig+".png", bbox_inches='tight')
        plt.close()
    else:
        fig.show()










