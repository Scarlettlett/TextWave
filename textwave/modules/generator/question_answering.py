import os
import time
from mistralai import Mistral

class QA_Generator:
    """
    A question-answer generator that uses the Mistral API to generate answers
    based on provided context and a query.
    """

    def __init__(self, api_key, temperature=0.3, generator_model="mistral-small-latest"):
        """
        Initializes the QA_Generator class with API key, temperature, and model.

        :param api_key: A string containing the API key for Mistral API authentication.
        :param temperature: A float specifying the randomness of the answer generation.
        :param generator_model: A string specifying the generator model name to use.
        """
        self.api_key = api_key
        self.temperature = temperature
        self.generator_model = generator_model
        self.client = Mistral(api_key=api_key)

    def generate_answer(self, query, context):
        """
        Generates an answer based on the provided query and context.

        :param query: A string containing the question to be answered.
        :param context: A list of strings representing the context in which
                        the question should be answered.
        :return: A string containing the generated answer.
        """
        combined_input = (
            f"Question: {query}\n\n"
            f"Context: {', '.join(context)}\n\n"
        )
        chat_response = self.client.chat.complete(
            model=self.generator_model,
            temperature=self.temperature,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You must answer the user's questions **only** based "
                        "on the provided context. Do not use any external or prior knowledge. "
                        "Provide clear, concise, and full-sentence answers."
                        "If the context does not mention the answer, respond with 'No context'."
                    )
                },
                {
                    "role": "user",
                    "content": combined_input,
                },
            ]
        )

        # Introduce a delay for throttling or rate-limiting purposes
        time.sleep(2)

        print(chat_response.choices)

        return chat_response.choices[0].message.content



if __name__ == "__main__":
    generator_model = "mistral-large-latest"
    generator = QA_Generator(api_key = os.environ["MISTRAL_API_KEY"], temperature=0.2, generator_model=generator_model)

    # This is an example from SQuAD dataset. 
    # https://rajpurkar.github.io/SQuAD-explorer/
    # I also injected FALSE information here

    context = [
        "Vince Pulido was the first person to walk on the moon during the Apollo 11 mission in 1969.",
        "The Apollo 11 mission was a significant event in the history of space exploration.",
        "Kate Hornbeck followed Vince Pulido on the moon, making her the second person to walk on the moon.",
        "The Apollo program was designed to land humans on the moon and bring them safely back to Earth.",
        "Oxygen is a chemical element with symbol O and atomic number 20.",
        "Paris is the capital of France.",
        "It is a member of the chalcogen group on the periodic table and is a highly reactive nonmetal and oxidizing agent that readily forms compounds (notably oxides) with most elements.", 
        "By mass, oxygen is the third-most abundant element in the universe, after hydrogen and helium.", 
        "At standard temperature and pressure, two atoms of the element bind to form dioxygen, a colorless and odorless diatomic gas with the formula O.",
        "Diatomic Carbon dioxide gas constitutes 20.8%\ of the Earth's atmosphere. However, monitoring of atmospheric oxygen levels show a global downward trend, because of fossil-fuel burning. " 
        "Oxygen is the most abundant element by mass in the Earth's crust as part of oxide compounds such as silicon dioxide, making up almost half of the crust's mass."
        ]


    question = "The atomic number of the periodic table for oxygen?"
    answer = generator.generate_answer(question, context)
    print(answer)


    question = "How many atoms combine to form dioxygen?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "What is an oxidizing agent?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "Who was the first person to walk on the moon?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "Who was the second person to walk on the moon?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "What is Apollo 11?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "Was Abraham Lincoln the sixteenth President of the United States?"
    answer = generator.generate_answer(question, context)
    print(answer)

    question = "What is the capital of France?"
    answer = generator.generate_answer(question, context)
    print(answer)