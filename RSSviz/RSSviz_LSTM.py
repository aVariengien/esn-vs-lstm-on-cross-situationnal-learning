from RSSviz import RecurentStateSpaceVisualisation
import sys
sys.path.append(r'..')
import numpy as np
import CSL_LSTM as lstm
import matplotlib.pyplot as plt
import keras

import importlib

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def save_open_figures(filename, figs=None, format = 'png', dpi = 100):

    if format == 'pdf':
        pp = PdfPages(filename)
        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        for fig in figs:
            fig.savefig(pp, format='pdf', dpi = dpi)
        pp.close()
    elif format == 'png':

        if figs is None:
            figs = [plt.figure(n) for n in plt.get_fignums()]
        id = 0
        for fig in figs:
            fig.savefig(filename + " Figure "+ str(id)+".png", format='png', dpi = dpi)
            id +=1
    else:
        print("invalid format")
##load the model


print("Loading the model ...")
name_id = '0.9144544314163248'
path = r"..\saved_LSTM\\"

model_for_test = keras.models.load_model(path+"best_model_for_test"+name_id+".hdf5")
model = keras.models.load_model(path+"best_model"+name_id+".hdf5")
print("Model loaded.")

## define the RSSviz

print("Creating the dimentional reduction ...")

rss_viz = RecurentStateSpaceVisualisation()

sent_to_use = lstm.test_sentences[:200] #we only use 200 sentence to speed up the process, use more sentence to get sharper details

all_states, object_nbs = lstm.get_inner_states(sent_to_use,
                                               model_for_test,
                                               only_last = False,
                                               info_obj_nb = True)


#Chosing wich part of the inner state of the LSTM to gather : cell: 2, state: 1, output: 0
#By default the study is done on the cell state
layer_id = 2

cells = all_states[layer_id]
unique_cells, unique_index = np.unique(cells, axis = 0, return_index = True) #get unique reservoir states

rss_viz.define_space(unique_cells)

print("Dimensional reduction defined.")


## Get the position ater reduction
points_2D = rss_viz.reducer.transform(unique_cells)


## Plot some sentences trajectories
sent = ['BEGIN a glass is on the middle and on the middle there is a red cup END', 'BEGIN on the right there is the blue orange and a orange cup is on the right END', 'BEGIN the cup on the left is green END']

intr = lstm.get_inner_states(sent, model_for_test)[layer_id]

save_sent_red_states = rss_viz.show_sentences(intr,
                                          sent,
                                          reduced_sentences_states = None,
                                          one_chunck = True,
                                          show_words = True,
                                          step_zero_included = True )

plt.scatter(points_2D[:,0], points_2D[:,1], alpha = 0.5, s = 10, c="yellow")
plt.show()


## RSSviz to know where are the first / second objects.


unique_cells, unique_indexes = np.unique(all_states[layer_id],
                                         axis = 0,
                                         return_index = True)

object_nbs = object_nbs[unique_indexes]


fig, ax = plt.subplots()
cax  = ax.scatter(points_2D[:,0], points_2D[:,1], c =object_nbs,cmap = plt.cm.get_cmap('tab10', 4),  alpha = 0.9, s = 10)

ax.text( save_sent_red_states[1][0], save_sent_red_states[1][1], "BEGIN")
ax.set_title("The place of the first and second objects clauses")

cbar  = fig.colorbar(cax, ticks=range(4),  label='Sentence position')

cbar.ax.set_yticklabels(['First object clause', """ 'and' word """, 'Second object clause', 'Final word'])
fig.show()


## We shuffle the sentences to easily get random data sample

sent_shuf = lstm.test_sentences[::]
np.random.shuffle(sent_shuf)


## Get the activations


activations = lstm.get_inner_states(sent_to_use, model_for_test)[0]
activations = np.unique(activations, axis = 0)


## Plot the repartition of activation distribution

for id_hist in range(0,lstm.nb_concepts*2,5): #go through 3 output neurons (22 in total)
    plt.figure()
    plt.title( "Repartition of the activation of" +  lstm.output_id_to_concept_dict[id_hist%11]+" object "+ str(id_hist//11 +1) )
    plt.hist(activations[:,id_hist], bins = 100)
    plt.show()


## Plot the output activations in the RSSvis
points_2D_non_unique = rss_viz.reducer.transform(all_states[layer_id])
activations = all_states[0]


for id_search in range(6,lstm.nb_concepts*2,5): #go through 3 output neurons (22 in total)
    plt.figure()
    plt.scatter(points_2D_non_unique[:,0],
                points_2D_non_unique[:,1],
                alpha = 0.9,c = activations[:, id_search],
                cmap='inferno',
                s=10,
                vmin=-0.3,
                vmax=1.3)
    plt.colorbar(cmap='inferno')
    plt.title("Activation of "+lstm.output_id_to_concept_dict[id_search%11]+" object "+ str(id_search//11 +1))
    plt.show()




