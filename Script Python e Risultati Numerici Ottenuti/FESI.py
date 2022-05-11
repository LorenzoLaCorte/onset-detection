#!/usr/bin/env python

# librerie principali
from __future__ import division
import sys
import librosa.display
import librosa
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import IPython
from statistics import mean
import warnings
import scipy
warnings.filterwarnings("ignore")


durations_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120]

lib_examples = ["brahms", "choice", "fishin", "humpback", "libri1", "libri2",
                "libri3", "nutcracker", "pistachio", "robin", "sweetwaltz", "trumpet", "vibeace"]

best = []
worst = []
list_avg = []
med_avg = []

for DURATION in durations_list:

    list_comparison_perc = []
    list_diff_onset = []
    list_mean_comparison = []

    filename = "Results" + str(DURATION) + ".dat"

    sys.stdout = open(filename, 'w')

    for i in range(len(lib_examples)):

        print("\n\n\n-----------------------------------------------------------")
        print("\nTraccia numero " + str(i) + ": " + str(lib_examples[i]))
        print("\n-----------------------------------------------------------")

        filename = librosa.example(lib_examples[i])

        y, sr = librosa.load(filename, duration=DURATION)

        # definisco i parametri principali
        hop_length = 512

        librosa_onset_frames = librosa.onset.onset_detect(
            y, sr=sr, hop_length=hop_length)
        librosa_onset_times = librosa.onset.onset_detect(
            y, sr=sr, units='time')

        def implemented_onset_strength(S, sr, hop_length):
            odf_sf = []

            norm = np.linalg.norm(S.ravel(), ord=2)

            for n in range(0, S.shape[1]-1):
                sum = 0

                for k in range(0, S.shape[0]-1):

                    sum += abs(S[k, n]) - abs(S[k, n-1])

                odf_sf.append(sum/norm)

            # Applico la half-wave rectifier function eliminando i negativi
            odf_sf = np.maximum(0.00, odf_sf)

            return odf_sf

        def implemented_peak_pick(odf_sf, w1, w2, w3, w4, delta, w5):
            onsets = []
            isPeak = True  # se resta True allora è un picco

            n_last_onset = 0

            # scorro tutti i frame di ODF(n)
            for n, odf_value in enumerate(odf_sf):

                if (n-w1) < 0 or (n-w3) < 0:
                    isPeak = False
                    continue

               # un odf_value ODF(n) è definito picco se ci sono tre condizioni:

                # 1. se ODF(n) = odf_value = max(ODF(n-w1:n+w2))
                if odf_value != max(odf_sf[n-w1:n+w2]):
                    isPeak = False

                # 2. se ODF(n) = odf_value >= mean(ODF(n-w3:n+w4)) + delta
                if odf_value < (mean(odf_sf[n-w3:n+w4]) + delta):
                    isPeak = False

                # 3. se n > (n_last_onset) + w5
                if(n < (n_last_onset + w5)):
                    isPeak = False

                if isPeak:
                    onsets.append(n)
                    n_last_onset = n

                isPeak = True  # resetto la variabile per il prossimo ciclo

            return onsets

        def my_onset_detect(y, sr=22050):

            if y is None:
                raise ParameterError("y must be provided")

            # 1. Preprocessing: nullo per ora*

            # 2. ODF: spectral flux

            n_mels = 138
            fmin = 27.5
            fmax = 16000.

            global S
            S = librosa.feature.melspectrogram(
                y, sr=sr, hop_length=hop_length, fmin=fmin, fmax=fmax, n_mels=n_mels)

            # onset detection function based on spectral flux
            global odf_sf
            odf_sf = implemented_onset_strength(S, sr, hop_length)

            # 3. Peak-Peaking based on spectral flux

            # pre_max: 3 valore ottimale, librosa gli da 1
            w1 = 3

            # post_max: 0 valore ottimale online, mentre 3 offline, librosa gli da 1
            w2 = 1

            # pre_avg: valore ottimale tra 4 e 12, librosa gli da 4
            # 4 è meno inclusivo, 12 è molto piu' inclusivo e aiuta a considerare l'ultima parte
            w3 = 4

            # post_avg: 0 valore ottimale online, mentre 1 offline, librosa gli da 1
            w4 = 1

            # è la soglia di threshold, librosa gli da 0.07
            delta = 0.003

            # wait: 3 valore ottimale online, mentre 0 offline, librosa gli da 1
            w5 = 3

            onsets = implemented_peak_pick(odf_sf, w1, w2, w3, w4, delta, w5)

            return onsets

        onset_frames = my_onset_detect(y, sr=sr)

        onset_times = librosa.core.frames_to_time(
            onset_frames, sr=sr, hop_length=512, n_fft=1024)

        # In[13]:

        n_lib = len(librosa_onset_frames)
        n_imp = len(onset_frames)

        print("Numero di Onset di Librosa: \n" + str(n_lib))
        print("Numero di Onset dell'Implementazione: \n" + str(n_imp))

        diff_onset = abs((n_lib - n_imp) / max(n_lib, n_imp))*100
        print("Differenza in Percentuale: \n" +
              str(diff_onset) + "%")

        list_diff_onset.append(diff_onset)

        # ## Confronto tra Array di Onset

        # In[14]:

        librosa_onset_frames = list(librosa_onset_frames)
        print("\nLibrosa: \n" + str(librosa_onset_frames))
        print("Implementazione: \n" + str(onset_frames))

        # ## Confronto tra le Medie degli Array di Onset

        # In[15]:

        m1 = np.mean(librosa_onset_frames)
        m2 = np.mean(onset_frames)
        n_frame = librosa.time_to_frames(
            DURATION, sr=22050, hop_length=hop_length, n_fft=1024)

        mean_comparison = abs(m1-m2)
        list_mean_comparison.append(mean_comparison)
        print("\nConfronto tra le Medie degli Array di Onset: " +
              str(mean_comparison))
        # print("Pesata sul numero di frame: " + str(mean_comparison/n_frame*100) + "%")

        # ## Confronto della Media della Distanza tra Onset Vicini

        # In[16]:

        def comp_near_frames(l1, l2, how_much_near):

            max_len = max(len(l1), len(l2))

            imp = 0
            lib = 0
            comp = 0
            acc = 0

            while(imp < (len(l2)-1) and lib < (len(l1)-1)):
                diff = l1[lib] - l2[imp]

                if(abs(diff) < how_much_near):
                    # print(str(l1[lib])+ " vs. " + str(l2[imp])+"\n\n")
                    acc += abs(diff)
                    lib += 1
                    imp += 1
                    comp += 1

                # se la diff è negativa allora librosa ha un frame molto minore, provo a portare avanti solo il suo indice
                elif(diff < 0):
                    lib += 1

                elif(diff > 0):
                    imp += 1

            if(comp == 0):
                return how_much_near
            return (acc/comp)

        how_much_near = 10
        comparison = comp_near_frames(
            librosa_onset_frames, onset_frames, how_much_near)
        comparison_perc = comparison/how_much_near*100
        
        print("\nMedia dei confronti tra onset (in frame) vicini " +
              str(how_much_near) + ": " + str(comparison))
        print("In percentuale con il fattore di vicinanza per la comparazione: " +
              str(comparison_perc) + "%")

        list_comparison_perc.append(comparison_perc)

    sys.stdout.close()

    sys.stdout = open("GeneralResults.dat", 'a')


    max_diff_onset = max(list_diff_onset)
    min_diff_onset = min(list_diff_onset)
    avg_diff_onset = 0 if len(list_diff_onset) == 0 else sum(
        list_diff_onset)/len(list_diff_onset)
    avg_med_onset = 0 if len(list_diff_onset) == 0 else sum(
		list_mean_comparison)/len(list_mean_comparison)

    max_mean_comparison = max(list_mean_comparison)
    min_mean_comparison = min(list_mean_comparison)
    avg_mean_comparison = 0 if len(list_mean_comparison) == 0 else sum(
        list_mean_comparison)/len(list_mean_comparison)

    max_comparison_perc = max(list_comparison_perc)
    min_comparison_perc = min(list_comparison_perc)
    avg_comparison_perc = 0 if len(list_comparison_perc) == 0 else sum(
        list_comparison_perc)/len(list_comparison_perc)

    worst.append(str(lib_examples[list_diff_onset.index(max_diff_onset)]))
    best.append(str(lib_examples[list_diff_onset.index(min_diff_onset)]))
    list_avg.append(avg_diff_onset)
    med_avg.append(avg_med_onset)

    print("\nDURATION=" + str(DURATION))

    print("\nLista delle differenze in numero di onset rispetto a Librosa: " +
          str(list_diff_onset))
    print("Lista delle differenze tra le medie degli onset rispetto a Librosa: " +
          str(list_mean_comparison))

    print("\nTraccia Peggiore sulla differenza in numero di onset rispetto a Librosa: " +
          str(lib_examples[list_diff_onset.index(max_diff_onset)]))
    print("Traccia Peggiore sulla differenza tra le medie degli onset rispetto a Librosa: " +
          str(lib_examples[list_mean_comparison.index(max_mean_comparison)]))
    #print("Traccia Peggiore sulla vicinanza di onset rispetto a Librosa: " +
    #      str(lib_examples[list_comparison_perc.index(max_comparison_perc)]))

    print("\nTraccia Migliore sulla differenza in numero di onset rispetto a Librosa: " +
          str(lib_examples[list_diff_onset.index(min_diff_onset)]))
    print("Traccia Migliore sulla differenza tra le medie degli onset rispetto a Librosa: " +
          str(lib_examples[list_mean_comparison.index(min_mean_comparison)]))
    #print("Traccia Migliore sulla vicinanza di onset rispetto a Librosa: " +
    #      str(lib_examples[list_comparison_perc.index(min_comparison_perc)]))

    print("\nMedia Generale sulla differenza in numero di onset rispetto a Librosa: " +
          str(avg_diff_onset) + "%")
    print("Media Generale sulla differenza tra le medie degli onset rispetto a Librosa: " +
          str(avg_mean_comparison))
    #print("Media Generale sulla vicinanza di onset rispetto a Librosa: " +
    #      str(avg_comparison_perc) + "%")

    sys.stdout.close()


sys.stdout = open("GeneralResults.dat", 'a')

print("\n\nLista delle migliori traccie sul numero di onset rispetto a Librosa: " +
  str(worst))
print("\nLista delle migliori traccie sul numero di onset rispetto a Librosa: " +
  str(best))
print("\nLista delle medie generali sulle differenze in numero di onset rispetto a Librosa: " +
  str(list_avg))
print("\nLista delle medie generali sulle differenze tra le medie degli onset rispetto a Librosa: " +
  str(med_avg))


sys.stdout.close()


