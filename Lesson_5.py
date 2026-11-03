import os
import re

from random import shuffle
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
from utils import *


## ------------------------------------------------------ ##
load_dotenv("../.env")

client = OpenAI(api_key = os.environ["OPENAI_API_KEY"])

pb_client = OpenAI(base_url = os.environ["PREDIBASE_MODEL_LLAMA_URL"],
                    api_key = os.environ["PREDIBASE_API_KEY"], )

## ------------------------------------------------------ ##
ds = load_dataset("mrSoul7766/ECTSum")
transcript = ds["train"][1]["text"]
print(transcript[ : 1983])

## ------------------------------------------------------ ##
SUMMARIZE_PROMPT = """
                    Generate a concise summary of the information in the following earnings call
                    transcript.

                    Only respond with the summary, do not include any extraneous text.

                    Transcript:

                    {transcript}
                    """

def summarize(transcript, n = 1):
    prompt = SUMMARIZE_PROMPT.format(transcript = transcript)

    messages = [{"role" : "user", "content" : prompt}, ]

    return pb_client.chat.completions.create(model = MODEL_NAME,
                                            messages = messages,
                                            n = n,
                                            temperature = 0.9, )

## ------------------------------------------------------ ##
resp = summarize(transcript)
summary = resp.choices[0].message.content
print(summary)

## ------------------------------------------------------ ##
JUDGE_PROMPT_V1 = """
                    Rate the following summary of an earnings call transcript on a scale from 1
                    to 10.

                    1 means the summary is very poor, 10 means the summary is very good.

                    Provide reasoning followed by the final score at the end surrounded by
                    <score> tags.

                    For example:

                    <score>1</score>

                    Transcript:

                    {transcript}

                    Summary:

                    {summary}
                    """

def judge_reward_v1(transcript: str, summary: str, model: str = "gpt-4o-mini",
                    verbose: bool = False, ) -> float:
    prompt = JUDGE_PROMPT_V1.format(transcript = transcript, summary = summary, )

    messages = [{"role" : "user", "content" : prompt}, ]

    resp = client.chat.completions.create(model = model,
                                        messages = messages,
                                        n = 1,
                                        temperature = 0, )

    completion = resp.choices[0].message.content

    if verbose:
        print(completion)

    try:
        match = re.search(r"<score>(\d+)<\/score>", completion)

        if match is None:
            return 0

        score = match.group(1).strip()
        score = int(score)

    except:
        score = 0

    return score / 10

## ------------------------------------------------------ ##
score = judge_reward_v1(transcript, summary, verbose = True)
print(score)

## ------------------------------------------------------ ##
resp = summarize(transcript, n = 8)
summaries = [choice.message.content for choice in resp.choices]

## ------------------------------------------------------ ##
scores = [judge_reward_v1(transcript, summary) for summary in summaries]
scores

## ------------------------------------------------------ ##
QUIZ_PROMPT = """
                Generate a multiple-choice quiz based on the information in the following earnings
                call transcript.

                Example:

                ```
                1. What was the q1 adjusted earnings per share?
                a) $3.34
                b) $5.32
                c) $2.49
                d) $7.78

                2. By what percent did same store sales rise in q1?
                a) 29.4%
                b) 32.1%
                c) 24.7%
                d) 21.2%

                ===== ANSWERS =====
                1. a
                2. c
                ```

                Limit the length of the quiz to the top 10 most relevant questions for financial
                analysts.

                Transcript:

                {transcript}
                """

## ------------------------------------------------------ ##
class Question(BaseModel):
    text: str
    options: list[str]
    answer: int

    def shuffle_options(self) -> None:
        correct = self.options[self.answer]

        shuffled = self.options.copy()
        shuffle(shuffled)

        self.options = shuffled
        self.answer = shuffled.index(correct)

    def __str__(self) -> str:
        output = [self.text]

        for i, option in enumerate(self.options):
            output.append(f"{chr(65 + i)}. {option}")

        return "\n".join(output)


class Quiz(BaseModel):
    questions: list[Question]

    def shuffle_all_questions(self) -> None:
        for question in self.questions:
            question.shuffle_options()

    def __str__(self) -> str:
        output = []

        for i, question in enumerate(self.questions, 1):
            output.append(f"\nQuestion {i}: ")
            output.append(str(question))

        return "\n".join(output)


def create_quiz(transcript: str):
    prompt = QUIZ_PROMPT.format(transcript = transcript)

    messages = [{"role" : "user", "content" : prompt}, ]

    resp = client.beta.chat.completions.parse(model = "gpt-4o-mini",
                                            messages = messages,
                                            temperature = 0.7,
                                            response_format = Quiz, )

    quiz = resp.choices[0].message.parsed
    quiz.shuffle_all_questions()

    return quiz

## ------------------------------------------------------ ##
quiz = create_quiz(transcript)
print(quiz)

## ------------------------------------------------------ ##
letter_to_index = {"A" : 0, "B" : 1, "C" : 2, "D" : 3}
index_to_letter = ["A", "B", "C", "D"]

TAKE_QUIZ_PROMPT = """
                    Use the provided summary of a transcript to answer the following quiz.

                    Quiz:

                    {quiz}

                    Summary:

                    {summary}

                    Respond with just a list of answers and no additional text, for example:

                    [A, D, C, B, B, C, D, A, A, B]

                    You must provide an answer for all 10 questions. If you don't know the answer,
                    answer with "0" for that question.

                    Example:

                    [A, D, 0, B, B, C, D, A, A, B]
                    """

def take_quiz(summary, quiz):
    question_strs = []

    for question in quiz.questions:
        question_str = question.text

        for i, option in enumerate(question.options):
            letter = index_to_letter[i]
            question_str += f"\n{letter}. {option}"

        question_strs.append(question_str)

    quiz_str = "\n\n".join(question_strs)

    prompt = TAKE_QUIZ_PROMPT.format(quiz = quiz_str, summary = summary)

    resp = client.chat.completions.create(model = "gpt-4o-mini",
                                        messages = [{"role" : "user", "content" : prompt}],
                                        temperature = 0, )

    resp_str = resp.choices[0].message.content

    answers = resp_str.strip('[]').split(', ')

    return answers

## ------------------------------------------------------ ##
answers = take_quiz(summaries[0], quiz)
answers

## ------------------------------------------------------ ##
def score_quiz_answers(answers, quiz):
    assert len(answers) == len(quiz.questions)

    total = len(answers)
    correct = 0

    for answer, question in zip(answers, quiz.questions):
        expected_answer = index_to_letter[question.answer]

        if answer == expected_answer:
            correct += 1

    return correct / total

## ------------------------------------------------------ ##
score_quiz_answers(answers, quiz)

## ------------------------------------------------------ ##
def print_quiz_table(all_answers, rewards):
    advantages = compute_advantages(rewards)
    length = len(all_answers)
    elems = list(zip(range(length), rewards, advantages))

    headers = ["Index", "Reward", "Advantage"]
    table = tabulate(elems, headers = headers, tablefmt = "grid").split("\n")

    for row in table:
        print(row)

## ------------------------------------------------------ ##
all_answers = []
quiz_rewards = []

for summary in summaries:
    answers = take_quiz(summary, quiz)
    all_answers.append(answers)
    quiz_rewards.append(score_quiz_answers(answers, quiz))

## ------------------------------------------------------ ##
print_quiz_table(all_answers, quiz_rewards)
