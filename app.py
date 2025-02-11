import pickle

import streamlit as st


def validate_sequence(sequence):
    """Validate if the sequence contains only ACGUT nucleotides."""
    valid_chars = set("ACGUT")
    sequence = sequence.upper()
    return all(char in valid_chars for char in sequence)


def process_text(
    input_text, tract_model, tract_model_refined, loop_model, loop_model_refined
):
    input_text = input_text.upper()
    if not validate_sequence(input_text):
        raise ValueError("Sequence must contain only A, C, G, U, or T nucleotides")

    observation = []

    for c in input_text:
        if c == "A":
            observation.append(0)
        elif c == "C":
            observation.append(1)
        elif c == "G":
            observation.append(2)
        elif c == "T" or c == "U":
            observation.append(3)
        else:
            raise ValueError(f"Invalid character {c}")

    tract_states, tract_scores = tract_model.decode([observation])
    tract_refined_states, tract_refined_scores = tract_model_refined.decode(
        [observation]
    )
    loop_states, loop_scores = loop_model.decode([observation])
    loop_refined_states, loop_refined_scores = loop_model_refined.decode([observation])

    return (
        "Tract Model States: " + str(tract_states),
        "Tract Model Scores: " + str(tract_scores),
        "Refined Tract Model States: " + str(tract_refined_states),
        "Refined Tract Model Scores: " + str(tract_refined_scores),
        "Loop Model States: " + str(loop_states),
        "Loop Model Scores: " + str(loop_scores),
        "Refined Loop Model States: " + str(loop_refined_states),
        "Refined Loop Model Scores: " + str(loop_refined_scores),
    )


def main():
    with open("tract_model.pkl", "rb") as f:
        tract_model = pickle.load(f)
    with open("tract_model_refined.pkl", "rb") as f:
        tract_model_refined = pickle.load(f)
    with open("loop_model.pkl", "rb") as f:
        loop_model = pickle.load(f)
    with open("loop_model_refined.pkl", "rb") as f:
        loop_model_refined = pickle.load(f)

    st.title("G4 loop location prediction")

    # Create text input
    user_input = st.text_input("Enter your G4 sequence:")

    # Create button
    if st.button("Process"):
        if user_input:
            try:
                # Process the input and display result
                results = process_text(
                    user_input,
                    tract_model,
                    tract_model_refined,
                    loop_model,
                    loop_model_refined,
                )
                for title, result in zip(
                    [
                        "Tract Model States",
                        "Tract Model Scores",
                        "Refined Tract Model States",
                        "Refined Tract Model Scores",
                        "Loop Model States",
                        "Loop Model Scores",
                        "Refined Loop Model States",
                        "Refined Loop Model Scores",
                    ],
                    results,
                ):
                    st.subheader(title)
                    st.code(result, language=None)
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Please enter some text")


if __name__ == "__main__":
    main()
