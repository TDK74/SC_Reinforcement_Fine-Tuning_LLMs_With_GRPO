import os
import re

from dataclasses import dataclass
from enum import Enum
from typing import List
from dotenv import load_dotenv
from openai import OpenAI
from transformers import AutoTokenizer


## ------------------------------------------------------ ##
_ = load_dotenv()

client = OpenAI(
                base_url = os.environ["PREDIBASE_MODEL_QWEN_URL"],
                api_key = os.environ["PREDIBASE_API_KEY"],
                )

## ------------------------------------------------------ ##
base_model_id = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

## ------------------------------------------------------ ##
SYSTEM_PROMPT = """
                You are playing Wordle, a word-guessing game.

                ### Game Rules:
                - You have **6 tries** to guess a secret **5-letter** word.
                - Each guess must be a valid **5-letter English word**.
                - After each guess, you will receive feedback indicating how close
                your guess was.

                ### Feedback Format:
                Each letter in your guess will receive one of three symbols:
                1. ✓ : The letter is in the word and in the CORRECT position.
                2. - : The letter is in the word but in the WRONG position.
                3. x : The letter is NOT in the word.

                ### Example:
                Secret Word: BRISK

                Guess 1: STORM → Feedback: S(-) T(x) O(x) R(-) M(x)
                Guess 2: BRAVE → Feedback: B(✓) R(✓) A(x) V(x) E(x)
                Guess 3: BRISK → Feedback: B(✓) R(✓) I(✓) S(✓) K(✓)

                ### Response Format:
                Think through the problem and feedback step by step. Make sure to
                first add your step by step thought process within <think> </think>
                tags. Then, return your guessed word in the following format:
                <guess> guessed-word </guess>.
                """

## ------------------------------------------------------ ##
class LetterFeedback(Enum):
    CORRECT = "✓"
    WRONG_POS = "-"
    WRONG_LETTER = "x"


@dataclass
class GuessWithFeedback:
    guess: str
    feedback: List[LetterFeedback]

    def __repr__(self) -> str:
        feedback_str = " ".join(f"{letter}({fb.value})" for letter, fb in zip(self.guess,
                                                                            self.feedback))

        return f"{self.guess} → Feedback: {feedback_str}"

## ------------------------------------------------------ ##
def render_user_prompt(past_guesses: List[GuessWithFeedback]) -> str:
    prompt = "Make a new 5-letter word guess."

    if past_guesses:
        prompt += "\n\nHere is some previous feedback:"

        for i, guess in enumerate(past_guesses):
            prompt += f"\nGuess {i + 1}: {guess}"

    return prompt

## ------------------------------------------------------ ##
def render_prompt(past_guesses: List[GuessWithFeedback]):
    messages = [
                {
                "role" : "system",
                "content" : SYSTEM_PROMPT
                },
                {
                "role" : "user",
                "content" : render_user_prompt(past_guesses)
                },
                {
                "role" : "assistant",
                "content" : "Let me solve this step by step.\n<think>"
                }
            ]

    return tokenizer.apply_chat_template(
                                        messages, tokenize = False,
                                        continue_final_message = True
                                        )

## ------------------------------------------------------ ##
def generate_stream(prompt: str, adapter_id: str = "") -> str:
    response = client.completions.create(
                                        model = adapter_id,
                                        prompt = prompt,
                                        temperature = 0.0,
                                        max_tokens = 2048,
                                        stream = True,
                                        )

    completion = ""

    for chunk in response:
        if chunk.choices[0].text is not None:
            content = chunk.choices[0].text
            print(content, end = "", flush = True)
            completion += content

    print()

    return completion

## ------------------------------------------------------ ##
secret_word = "CRAFT"

past_guesses = [
                GuessWithFeedback(
                                "CRANE", [
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.WRONG_LETTER,
                                        LetterFeedback.WRONG_LETTER,
                                ]),
                GuessWithFeedback(
                                "CRASH", [
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.CORRECT,
                                        LetterFeedback.WRONG_LETTER,
                                        LetterFeedback.WRONG_LETTER,
                                ]),
            ]

prompt = render_prompt(past_guesses)
print(prompt)

## ------------------------------------------------------ ##
base_completion = generate_stream(prompt)

## ------------------------------------------------------ ##
ft_completion = generate_stream(prompt, adapter_id = "wordle-dlai/2")

## ------------------------------------------------------ ##
def get_feedback(guess: str, secret_word: str) -> List[LetterFeedback]:
    valid_letters = set(secret_word)
    feedback = []

    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            feedback.append(LetterFeedback.CORRECT)

        elif letter in valid_letters:
            feedback.append(LetterFeedback.WRONG_POS)

        else:
            feedback.append(LetterFeedback.WRONG_LETTER)

    return feedback

## ------------------------------------------------------ ##
def next_turn(
            past_guesses: List[GuessWithFeedback],
            secret_word: str,
            adapter_id = ""
            ):
    prompt = render_prompt(past_guesses)
    completion = generate_stream(prompt, adapter_id)
    match = re.search(
                    r"<guess>\s*(.*?)\s*</guess>", completion, re.DOTALL
                    )

    if not match:
        raise RuntimeError("invalid guess")

    guess = match.group(1).upper()
    feedback = get_feedback(guess, secret_word)
    past_guesses.append(GuessWithFeedback(guess, feedback))
    print("\n\n")
    print(("-" * 100) + "\n")

    for past_guess in past_guesses:
        print(past_guess)

    if guess == secret_word:
        print("🎉 SUCCESS 🎉")

    elif len(past_guesses) >= 6:
        print("❌ better luck next time... ❌")

## ------------------------------------------------------ ##
secret_word = "BRICK"
past_guesses = []
adapter_id = ""

## ------------------------------------------------------ ##
next_turn(past_guesses, secret_word, adapter_id)

## ------------------------------------------------------ ##
secret_word = "BRICK"
past_guesses = []
adapter_id = "wordle-dlai/2"

## ------------------------------------------------------ ##
next_turn(past_guesses, secret_word, adapter_id)
