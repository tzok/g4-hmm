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

    result = []

    for algorithm in ("viterbi", "map"):
        _, tract_states = tract_model.decode([observation], algorithm=algorithm)
        _, loop_states = loop_model.decode([observation], algorithm=algorithm)
        result.append(
            input_text
            + "\n"
            + "".join(["." if s == 0 else "T" for s in tract_states])
            + "\n"
            + "".join(["." if s == 0 else "L" for s in loop_states])
        )

    return result


def main():
    with open("tract_model.pkl", "rb") as f:
        tract_model = pickle.load(f)
    with open("tract_model_refined.pkl", "rb") as f:
        tract_model_refined = pickle.load(f)
    with open("loop_model.pkl", "rb") as f:
        loop_model = pickle.load(f)
    with open("loop_model_refined.pkl", "rb") as f:
        loop_model_refined = pickle.load(f)

    st.title("HMM for G4 tract & loop location prediction")

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
                        "Viterbi",
                        "MAP",
                    ],
                    results,
                ):
                    st.subheader(title)
                    st.code(result, language=None)
            except ValueError as e:
                st.error(str(e))
        else:
            st.warning("Enter your G4 sequence")


if __name__ == "__main__":
    main()
