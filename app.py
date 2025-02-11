import streamlit as st

from hmm import train_hmm_for_dataset


def validate_sequence(sequence):
    """Validate if the sequence contains only ACGUT nucleotides."""
    valid_chars = set("ACGUT")
    sequence = sequence.upper()
    return all(char in valid_chars for char in sequence)


def process_text(input_text, tract_model, loop_model):
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
    tract_model = train_hmm_for_dataset("tract")
    loop_model = train_hmm_for_dataset("loop")

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
                    loop_model,
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
