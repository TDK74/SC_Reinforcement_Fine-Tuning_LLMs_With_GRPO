import warnings
import numpy as np
import torch

from utils import *

## ------------------------------------------------------ ##
warnings.filterwarnings("ignore")

## ------------------------------------------------------ ##
create_deployment()

## ------------------------------------------------------ ##
model_id = "Qwen/Qwen2.5-7B-Instruct"

## ------------------------------------------------------ ##
def wordle_reward(guess: str, secret_word: str) -> int:
    if guess.upper() == secret_word.upper():
        return 1

    else:
        return 0

## ------------------------------------------------------ ##
secret_word = "POUND"

past_guesses = [
                GuessWithFeedback.from_secret(guess = "CRANE", secret = secret_word),
                GuessWithFeedback.from_secret(guess = "BLOND", secret = secret_word),
                GuessWithFeedback.from_secret(guess = "FOUND", secret = secret_word),
                ]
past_guesses

## ------------------------------------------------------ ##
response = generate(get_messages(past_guesses))[0]
guess = extract_guess(response)
reward = wordle_reward(guess, secret_word)

print(f"Guessed Word: {guess} -> Reward: {reward}")

## ------------------------------------------------------ ##
def compute_advantages(rewards: list):
    rewards = np.array(rewards)

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    if std_reward == 0:
        return [0] * len(rewards)

    advantages = (rewards - mean_reward) / std_reward

    return advantages.tolist()

## ------------------------------------------------------ ##
rewards = [0.0, 0.2, 0.4, 0.5, 0.5, 0.6, 0.8, 1.0]
compute_advantages(rewards)

## ------------------------------------------------------ ##
def render_guess_table(response, reward_fn):
    guesses = [extract_guess(guess) for guess in response]
    rewards = [reward_fn(guess, secret_word) for guess in guesses]
    print_guesses_table(guesses, rewards)

## ------------------------------------------------------ ##
print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses = 8)
render_guess_table(response, wordle_reward)

## ------------------------------------------------------ ##
def wordle_reward_partial_credit(guess: str, secret_word: str) -> float:
    if len(guess) != len(secret_word):
        return 0.0

    valid_letters = set(secret_word)
    reward = 0.0

    for letter, secret_letter in zip(guess, secret_word):
        if letter == secret_letter:
            reward += 0.2

        elif letter in valid_letters:
            reward += 0.1

        else:
            # no reward
            pass

    return reward

## ------------------------------------------------------ ##
print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses = 8, temperature = 0)
render_guess_table(response, wordle_reward_partial_credit)

## ------------------------------------------------------ ##
print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses = 8, temperature = 1.3)
render_guess_table(response, wordle_reward_partial_credit)

## ------------------------------------------------------ ##
print(f"Secret: {secret_word}")
response = generate(get_messages(past_guesses), num_guesses = 8, temperature = 0.7)
render_guess_table(response, wordle_reward_partial_credit)
