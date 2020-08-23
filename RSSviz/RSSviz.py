
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import umap


class RecurentStateSpaceVisualisation():
    def __init__(self, n_compo=2):
        self.reducer = umap.UMAP(n_components=n_compo, random_state = 42)

    def define_space(self, recurent_states):
        self.reducer.fit(recurent_states)

    def show_space(self, recurent_states, labels):
        """Plot the vector contained in recurent_states
        after dimension reduction with labels for each point"""
        plt.figure()
        reduced_states = self.reducer.transform(recurent_states)
        fig, ax = plt.subplots()
        ax.scatter(reduced_states[:,0],reduced_states[:,1] , s = 5)

        for i, label in enumerate(labels):
            print(i)
            ax.annotate(label, (reduced_states[i][0], reduced_states[i][1]))

    def show_sentences(self,
                       sentences_states,
                       sentences,
                       show_words = True,
                       one_chunck = False,
                       split_on_space = True,
                       reduced_sentences_states = None,
                       step_zero_included = False):
        """Plot the states in sentences_states as lines in the RSSviz.
           Arguments:
           sentences_states - list of vector corresponding to
                              the hidden states during the processing of each sentence.
           sentences - list of strings

           show_words - show the corresponding word near each point
           one_chunck - if True, process all the states in one chuck for the dimension
                        reduction (sentences_states needs to be a numpy array). If False,
                        each sentence has its states reduced separatly (sentences_states
                        needs to be a list of numpy array).
           split_on_space - Should the strings of sentences be splited on space to extract
                            the words.
           reduced_sentences_states - If show_sentences has already been applied on these
                                      sentences, you can reused the points computed to
                                      avoid the time taken by the dimension reduction.
           step_zero_included - If True, the first state should be the initial state of the
                                RNN and so no word will be plotted next to it."""
        fig = plt.figure()

        if split_on_space:
            words = [s.split(" ") for s in sentences]
        else:
            words =sentences

        save_reduced_sentences_states = []

        if one_chunck and reduced_sentences_states is None: ## all the sentences are transoformed in one chunck
            all_reduced_sentences_states = self.reducer.transform(sentences_states)
            index = 0
        if one_chunck and not(reduced_sentences_states is None):
            all_reduced_sentences_states = reduced_sentences_states
            index = 0

        for i in range(len(sentences)):

            if not(one_chunck):
                if reduced_sentences_states is None:
                    reduced_sentence_states = self.reducer.transform(sentences_states[i])
                    save_reduced_sentences_states.append(reduced_sentence_states)
                else:
                    reduced_sentence_states = reduced_sentences_states[i]
            else:
                if not(step_zero_included):
                    reduced_sentence_states = all_reduced_sentences_states[index: index+len(words[i])]
                else:
                    reduced_sentence_states = all_reduced_sentences_states[index: index+len(words[i])+1]

                index += len(words[i])
                if step_zero_included:
                    index +=1

            plt.plot(reduced_sentence_states[:,0],reduced_sentence_states[:,1])

            ax = fig.axes[0]
            if show_words:
                for j, word in enumerate(words[i]):
                    if step_zero_included:
                        ax.annotate(word, (reduced_sentence_states[j+1][0], reduced_sentence_states[j+1][1]))
                    else:
                        ax.annotate(word, (reduced_sentence_states[j][0], reduced_sentence_states[j][1]))

        if one_chunck:
            return all_reduced_sentences_states
        return save_reduced_sentences_states


