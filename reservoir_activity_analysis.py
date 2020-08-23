import numpy as np
import random as rd
import matplotlib.pyplot as plt
from plots import *
import CSL_ESN as esn

def get_top(nb, get_max, a):
    """Return the indexes of the nb higher values in the
       array a. If get_max if False, it's the nb lower"""

    if get_max:
        ind = np.argpartition(a, -nb)[-nb:]
    else:
        ind = np.argpartition(a, nb)[:nb]
    ind = np.flip(ind[np.argsort(a[ind])])
    return ind


def threshold( x, threshold_value = 0.0):
    if np.abs(x) < threshold_value:
        return 0.0
    else:
        return x

def print_word_influence_on_units(reservoir_units, N= 100):
    """print how much each word influence the activity of
    the units with indexes reservoir_units. The stats are computed
    on N sentences chosen at random from the test sentences"""
    all_words = list(esn.word2one_hot_id.keys())
    #all_words.remove("BEGIN")
    all_act = [ [] for i in range(len(all_words))]
    for s in rd.sample(esn.test_sentences,N):
        res_act , _ = esn.get_int_states(reservoir, [s], one_chunck = True)
        var = np.abs(np.diff(res_act, axis = 0))
        words = s.split(" ")
        for i in range(len(var)):
            su = np.sum(var[i][reservoir_units])
            all_act[ esn.word2one_hot_id[ words[i+1]]].append(su)

    var_means = []
    var_std = []
    for i in range(len(all_act)-1):
        if all_words[i] != "BEGIN":
            var_means.append(np.mean(all_act[i]))
            var_std.append(np.std(all_act[i]))
            print(all_words[i], " mean:", np.round(var_means[i], 4), " std:", np.round(var_std[i],4))


def get_concept_name(output_neuron_id):
    return esn.output_id_to_concept_dict[output_neuron_id%11][:-1]+str(output_neuron_id//11 +1)+">"

## load the reservoir

print("Loading the reservoir ...")
name_id = '0.8885270787313605'
path = r"saved_ESN\\"

W = np.load(path+"W"+name_id+".npy", allow_pickle = True)
W = W.item()

Wout = np.load(path+"Wout"+name_id+".npy", allow_pickle = True)
Win = np.load(path+"Win"+name_id+".npy", allow_pickle = True)


reservoir = esn.ESNOnline(lr = esn.leak,
                    W = W,
                    Win = Win,
                    Wout = Wout,
                    alpha_coef = esn.alpha_coef)

print("Reservoir loaded.")

print("----------------------------------")
## The winglet effect

print("Generating winglet effect plot ...")
cmap = matplotlib.cm.get_cmap('tab20')

plt.figure()
for i in range(Wout.shape[0]):
    plt.plot(np.abs(np.sort(Wout[i])),color = cmap(i%20))
plt.title("Absolute values of the ordered readout connection weights for the ESN")


plt.yticks( [i for i in range(0,35,5)])
plt.xticks( [i for i in range(0,1000,100)])
plt.xlabel("Rank of the reservoir unit (increasing order)")
plt.ylabel("Absolute value of the connection weight from \n the reservoir unit to the output neuron")
plt.grid()
plt.show()


print("Winglet effect plot generated.")
print("----------------------------------")
## Compute the top 10 most connected units to each output neurons

print("Gathering reservoir states ...")
reservoir_states, _ = esn.get_int_states(reservoir, esn.test_sentences, False, False)
print("Reservoir states gathered.")

print("----------------------------------")
high_connected_units = set()
counts_units = {}

for search_id in range(22):

    ind_top_max = get_top(5, True, Wout[search_id])
    ind_top_min = get_top(5, False, Wout[search_id])

    for i in ind_top_max:
        high_connected_units.add(i)
        if not i in counts_units:
            counts_units[i] = 1
        else:
            counts_units[i] += 1

    for i in ind_top_min:
        high_connected_units.add(i)
        if not i in counts_units:
            counts_units[i] = 1
        else:
            counts_units[i] += 1


## Identify polyvalent reservoir units
# i.e. the reservoir units that are in at least 5 top 10 most connected.

polyvalent_units = []
nb_poly =0
for x in counts_units.keys():
    if counts_units[x] >= 5:
        nb_poly +=1
        polyvalent_units.append(x)

print(nb_poly, " polyvalent reservoir units have been identified :")
for x in polyvalent_units:
    print("Reservoir unit #",x," found in ",counts_units[x]," top 10 of the most connected units.")

#test sentence to plot activations
s = 'BEGIN a green glass is on the left and that is a orange cup END'

res_act , _ = esn.get_int_states(reservoir, [s], one_chunck = False)

labels = ["(polyvalent unit)"]*esn.N #to add text to the plot legend

esn.plot_hidden_state_activation(s,
                                 [res_act[0]],
                                 state='reservoir',
                                 units_to_plot = list(polyvalent_units),
                                 plot_variation = True,
                                 plot_sum = False,
                                 add_to_legend = labels)


print("----------------------------------")
## Identify the units the most connected to a specific output neuron

search_id = 0 #the id of the output neuron (by default : corresponding to the concept <glass_object1>, try to change it !)
ind_top_max = get_top(10, True, np.abs(Wout[search_id]))

weights_labels = []

for i in range(Wout[search_id].shape[0]):
    weights_labels.append(", connection weight to"+get_concept_name(search_id)+ str(round(Wout[search_id][i], 1) ))

#plot the activation pattern of the reservoir units
#the most connected to the output neuron define by search_id (default 0 : <glass_object1>)
esn.plot_hidden_state_activation(s,
                                 [res_act[0]],
                                 state='reservoir',
                                 units_to_plot = list(ind_top_max),
                                 plot_variation = True,
                                 plot_sum = False,
                                 add_to_legend = weights_labels)


# how specific these reservoir units are reacting to the word "glass" ?
print("Variation in activation of the units the most connected to the concept  "+get_concept_name(search_id)+": ")
print_word_influence_on_units(ind_top_max)







