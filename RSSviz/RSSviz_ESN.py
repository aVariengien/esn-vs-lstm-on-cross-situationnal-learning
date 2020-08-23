
import os

from RSSviz import RecurentStateSpaceVisualisation
import sys
sys.path.append(r'..')
import CSL_ESN as esn
import numpy as np
from reservoirpy import ESNOnline
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.colors
import importlib


def save_open_figures(filename, figs=None, format = 'pdf', dpi = 100):

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


## load the reservoir

print("Loading the reservoir ...")
name_id = '0.8885270787313605'
path = r"..\saved_ESN\\"

W = np.load(path+"W"+name_id+".npy", allow_pickle = True)
W = W.item()

Wout = np.load(path+"Wout"+name_id+".npy", allow_pickle = True)
Win = np.load(path+"Win"+name_id+".npy", allow_pickle = True)


reservoir = ESNOnline(lr = esn.leak,
                    W = W,
                    Win = Win,
                    Wout = Wout,
                    alpha_coef = esn.alpha_coef)

print("Reservoir loaded.")

##define the rss_viz

print("Defining the dimensional reduction ...")
rss_viz = RecurentStateSpaceVisualisation()


sent_shuf = esn.test_sentences[::]  #we shuffle the sentences to get a random sample
                                    #from the dataset used for training
np.random.shuffle(sent_shuf)

sent_to_use =  sent_shuf[:200] #we only use 200 sentence to speed up the process, use more sentence to get sharper details

reservoir_states, _, object_nbs = esn.get_int_states(reservoir, sent_to_use, one_chunck = True, only_last = False, get_obj_nb = True)

unique_states = np.unique(reservoir_states, axis = 0) #get unique reservoir states
rss_viz.define_space(unique_states)

print("Dimensionnal reduction defined.")

## Create RSS visualisations

print("Generating RSSviz ...")

points_2D = rss_viz.reducer.transform(unique_states)


sent = ['BEGIN a glass is on the middle and on the middle there is a red cup END', 'BEGIN on the right there is the blue orange and a orange cup is on the right END', 'BEGIN the cup on the left is green END']



#plot the trajectory of 3 sentences and we rememer the position of the points to avoid to redo computation
intr, labs_small = esn.get_int_states(reservoir, sent, True, False)
save_sent_red_states = rss_viz.show_sentences(intr,
                                          sent,
                                          reduced_sentences_states = None,
                                          one_chunck = True,
                                          show_words = True,
                                          step_zero_included =False)
plt.scatter(points_2D[:,0], points_2D[:,1], alpha = 0.5, s = 10, c="yellow")
plt.title("Trajectories of 3 sentences in the RSSviz")
plt.show()


## Where are the final states ?

final_states, la = esn.get_int_states(reservoir, sent_to_use, True, True)
final_states_2D = rss_viz.reducer.transform(final_states)

sent = sent_shuf[:500]
intr, _ = esn.get_int_states(reservoir, sent, True, True)
vectorize_sent = [[s] for s in sent]
rss_viz.show_sentences(intr,
                   vectorize_sent,
                   one_chunck = True,
                   show_words = True,
                   split_on_space = False,
                   step_zero_included = False) #plot the sentence text were its final state is

plt.scatter(points_2D[:,0], points_2D[:,1], alpha = 0.9, s = 10)
plt.scatter(final_states_2D[:,0] , final_states_2D[:,1], s = 10)
plt.title("The position of the final states of sentences in the RSSviz")
plt.show()

## Where are the first / second objects ?


_ , unique_indexes = np.unique(reservoir_states, axis = 0, return_index = True)
object_nbs = object_nbs[unique_indexes] #get wich object each word relates to


fig, ax = plt.subplots()
cax  = ax.scatter(points_2D[:,0], points_2D[:,1], c =object_nbs,cmap = plt.cm.get_cmap('tab10', 4),  alpha = 0.9, s = 10)
ax.text( save_sent_red_states[0][0], save_sent_red_states[0][1], "BEGIN")

ax.set_title("The place of the first and second objects clauses")

cbar  = fig.colorbar(cax, ticks=range(4),  label='Sentence position')
cbar.ax.set_yticklabels(['First object clause', """ 'and' word """, 'Second object clause', 'Final word'])
fig.show()



## Get output activations

activations = []
for i in range(unique_states.shape[0]):
    activations.append(np.dot(Wout,unique_states[i]))
activations = np.array(activations)


## Plot hitsograms of the output activation distribution

for id_hist in range(0,esn.nb_concepts*2,5): #we go trough several arbitrary concepts
    plt.figure()
    plt.title( "Repartition of the activation of" +  esn.output_id_to_concept_dict[id_hist%11]+" object "+ str(id_hist//11 +1) )
    plt.hist(activations[:,id_hist], bins = 100)
    plt.show()

### Where are the errors ?


errors = esn.getErrorsInfo(reservoir_states, sent_to_use, Wout)

points_2D_non_unique = rss_viz.reducer.transform(reservoir_states)

fig, ax = plt.subplots()
cax  = ax.scatter(points_2D_non_unique[:,0],
                  points_2D_non_unique[:,1],
                  c =errors,
                  cmap = plt.cm.get_cmap('tab10', 9),
                  alpha = 0.9,
                  s = 10)

ax.text( save_sent_red_states[0][0], save_sent_red_states[0][1], "BEGIN")
ax.set_title("Where the errors occur")


cbar  = fig.colorbar(cax, ticks=range(9),  label='Objects errors (object 1, object 2)')

cbar.ax.set_yticklabels(
['Not valid, Not valid',
"Valid, Not valid",
'Exact, Not valid',
'Not valid, Valid',
'Valid, Valid',
'Exact, Valid',
'Not valid, Exact',
'Valid, Exact',
'Exact, Exact'])

fig.show()


## Plot the output activations on the RSSviz


for id_search in range(0,esn.nb_concepts*2,5):
    fig = plt.figure()
    plt.scatter(points_2D[:,0], points_2D[:,1], c=activations[:,id_search],cmap='inferno',vmin=-0.3, vmax=1.3, alpha = 0.9, s = 10)
    plt.colorbar(cmap='inferno')
    plt.title("Activation of "+esn.output_id_to_concept_dict[id_search%11]+" object "+ str(id_search//11 +1))
    plt.show()

