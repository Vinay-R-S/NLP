"""
NLP Program 7: Question Answering with Transformers
"""
from transformers import pipeline

def main():
    print("Loading Pretrained QA Model...")
    # Load Pretrained QA Model
    qa_pipeline = pipeline("question-answering")

    # Define Context (SQuAD style)
    context = """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
    especially computer systems. These processes include learning, reasoning, and self-correction.
    AI is widely used in applications such as natural language processing, robotics, and computer vision.
    """

    # Define Question
    question = "What are the processes included in AI?"

    print("Running QA inference...")
    # Get Answer
    result = qa_pipeline(question=question, context=context)

    # Display Output
    print("\n--- Results ---")
    print("Question:", question)
    print("Answer:", result['answer'])
    print("Confidence Score:", result['score'])

if __name__ == "__main__":
    main()
