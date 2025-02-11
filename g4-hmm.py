#! /usr/bin/env python
import glob
import pickle

import numpy as np
from hmmlearn import hmm


def load_loop_data():
    observations = []
    states = []

    for path in glob.iglob("qrs/*.dbn"):
        with open(path) as f:
            lines = f.readlines()
            sequence, loops = lines[0].strip(), lines[3].strip()

        observation = []
        flag = False

        for c in sequence:
            if c == "A":
                observation.append(0)
            elif c == "C":
                observation.append(1)
            elif c == "G":
                observation.append(2)
            elif c == "T" or c == "U":
                observation.append(3)
            elif c == "-":
                continue
            elif c == "N":
                flag = True
                break
            else:
                raise ValueError(f"Invalid character {c}")

        if flag:
            continue

        observations.append(np.array(observation))
        state = []

        for c in loops:
            if c == ".":
                state.append(0)
            elif c in "pPlLd":
                state.append(1)
            elif c == "-":
                continue
            else:
                raise ValueError(f"Invalid character {c}")

        states.append(np.array(state))

    return zip(observations, states)


def load_tract_data():
    observations = []
    states = []

    for path in glob.iglob("qrs/*.dbn"):
        with open(path) as f:
            lines = f.readlines()
            sequence, tract = lines[0].strip(), lines[1].strip()

        observation = []
        state = []
        flag = False

        for s, t in zip(sequence, tract):
            if s == "A":
                observation.append(0)
            elif s == "C":
                observation.append(1)
            elif s == "G":
                observation.append(2)
            elif s == "T" or s == "U":
                observation.append(3)
            elif s == "-":
                continue
            elif s == "N":
                flag = True
                break
            else:
                raise ValueError(f"Invalid character {s}")

            if t == "." or s != "G":
                state.append(0)
            elif t == "-":
                continue
            else:
                state.append(1)

        if flag:
            continue

        observations.append(np.array(observation))
        states.append(np.array(state))

    return zip(observations, states)


def train_hmm(paired_seqs, n_states=2, n_obs=4):
    # n_states = 2  # number of hidden states (0 and 1)
    # n_obs = 4  # number of possible emissions (0,1,2,3)

    # Initialize counts
    start_counts = np.zeros(n_states)
    trans_counts = np.zeros((n_states, n_states))
    emission_counts = np.zeros((n_states, n_obs))

    # Loop over each sequence to count occurrences
    for obs_seq, state_seq in paired_seqs:
        # Count start state
        start_counts[state_seq[0]] += 1

        # Count emissions for each position
        for s, o in zip(state_seq, obs_seq):
            emission_counts[s, o] += 1

        # Count transitions (if sequence has more than one state)
        for s_from, s_to in zip(state_seq[:-1], state_seq[1:]):
            trans_counts[s_from, s_to] += 1

    # Normalize counts to get probabilities
    startprob = start_counts / start_counts.sum()

    # For transition probabilities, normalize each state's row
    transmat = np.where(
        trans_counts.sum(axis=1, keepdims=True) == 0,
        np.full_like(trans_counts, 1.0 / n_states),
        trans_counts / trans_counts.sum(axis=1, keepdims=True),
    )

    # For emission probabilities, normalize each state's row
    emissionprob = np.where(
        emission_counts.sum(axis=1, keepdims=True) == 0,
        np.full_like(emission_counts, 1.0 / n_obs),
        emission_counts / emission_counts.sum(axis=1, keepdims=True),
    )

    # Create and initialize the HMM model
    model = hmm.CategoricalHMM(n_components=n_states, n_iter=100, init_params="")
    # The init_params="" argument tells the model not to override your parameter initialization.

    model.startprob_ = startprob
    model.transmat_ = transmat
    model.emissionprob_ = emissionprob
    return model


def refine_hmm(model, paired_seqs):
    # Prepare the observation sequences for training.
    # hmmlearn expects a single 2D array of shape (n_samples, 1) and a list of sequence lengths.
    obs_list = []
    lengths = []
    for obs_seq, _ in paired_seqs:
        # Reshape each observation sequence to (-1, 1)
        obs_seq_reshaped = obs_seq.reshape(-1, 1)
        obs_list.append(obs_seq_reshaped)
        lengths.append(len(obs_seq_reshaped))

    # Concatenate individual sequences into one long sequence.
    concatenated_obs = np.concatenate(obs_list)

    # Use the unsupervised Baum-Welch algorithm to refine the model’s parameters.
    # Note: The state labels are ignored here—only the observations are used.
    model.fit(concatenated_obs, lengths)


def train_and_refine_hmm(data):
    if data == "loop":
        paired_seqs = list(load_loop_data())
    elif data == "tract":
        paired_seqs = list(load_tract_data())
    else:
        raise ValueError(f"Invalid data type {data}")

    model = train_hmm(paired_seqs)

    # Print the parameters before unsupervised refinement.
    print("Parameters BEFORE unsupervised training:")
    print("Start probabilities:\n", model.startprob_)
    print("Transition matrix:\n", model.transmat_)
    print("Emission probability matrix:\n", model.emissionprob_)

    with open(f"{data}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Refine the model using unsupervised training.
    refine_hmm(model, paired_seqs)

    with open(f"{data}_model_refined.pkl", "wb") as f:
        pickle.dump(model, f)

    # Print the parameters after unsupervised refinement.
    print("\nParameters AFTER unsupervised training (fit):")
    print("Start probabilities:\n", model.startprob_)
    print("Transition matrix:\n", model.transmat_)
    print("Emission probability matrix:\n", model.emissionprob_)


def main():
    train_and_refine_hmm("loop")
    train_and_refine_hmm("tract")


if __name__ == "__main__":
    main()
