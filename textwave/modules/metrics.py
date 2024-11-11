from qa_metrics.transformerMatcher import TransformerMatcher
from qa_metrics.em import em_match


class Metrics_Automated:

    def __init__(self, model="distilbert"):
        # Supported models: roberta-large, tiny-bert, roberta, bert, distilbert, distilroberta
        self.transformer_matcher = TransformerMatcher(model)


    def exact_match(self, generated_answer, ground_truth_answer):
        match_result = em_match(ground_truth_answer, generated_answer)
        return match_result
    
    def transformer_match(self, generated_answer, ground_truth_answer, question):
        scores = self.transformer_matcher.get_scores(ground_truth_answer, generated_answer, question)
        match_result = self.transformer_matcher.transformer_match(ground_truth_answer, generated_answer, question)
        return scores, match_result



if __name__ == "__main__":
    metrics = Metrics_Automated()

    """Question: Who followed Vince Pulido to walk on the moon?"""
    
    
    # Correct answer example
    question = "Who was the first person to walk on the moon?"
    generated_answer = "Based on the provided context, Vince Pulido was the first person to walk on the moon."
    ground_truth_answer = "Vince Pulido"
    
    print("QUESTION:", question)
    print("Generated Answer:", generated_answer)
    print("True Answer:", ground_truth_answer)
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')
    print()

    question = "Who was the first person to walk on the moon?"
    generated_answer = "Based on the provided context, Pulido was the first person to walk on the moon."
    ground_truth_answer = "Vince Pulido"

    print("QUESTION:", question)
    print("Generated Answer:", generated_answer)
    print("True Answer:", ground_truth_answer)
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')
    print()


    # Incorrect answer example
    question = "Who was the first person to walk on the moon?"
    generated_answer = "Based on the provided context, Kate Hornbeck was the first person to walk on the moon."
    ground_truth_answer = "Vince Pulido"
    
    print("QUESTION:", question)
    print("Generated Answer:", generated_answer)
    print("True Answer:", ground_truth_answer)
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')
    print()


    # Correct answer example
    question = "How many atoms combine to form dioxygen?"
    generated_answer = "Based on the provided context, 2 atoms combine to form dioxygen."
    ground_truth_answer = "At standard temperature and pressure, two atoms of the element bind to form dioxygen."

    print("QUESTION:", question)
    print("Generated Answer:", generated_answer)
    print("True Answer:", ground_truth_answer)
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')
    print()


    # Incorrect answer example
    question = "How many atoms combine to form dioxygen?"
    generated_answer = "Based on the provided context, 5 atoms combine to form dioxygen."
    ground_truth_answer = "At standard temperature and pressure, two atoms of the element bind to form dioxygen."

    print("QUESTION:", question)
    print("Generated Answer:", generated_answer)
    print("True Answer:", ground_truth_answer)
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')
    print()