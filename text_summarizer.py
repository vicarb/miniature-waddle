from transformers import BartTokenizer, BartForConditionalGeneration
import torch

def summarize_text(text):
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, max_length=150, early_stopping=True)

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def main():
    # Replace this with the text you want to summarize
    text = """
    Certainly, I can create an essay about absurdism. Absurdism is a philosophical belief that emphasizes the irrationality and meaninglessness of human existence in a universe that is indifferent to our needs and desires. It is often associated with the works of French writer and philosopher Albert Camus, who explored the concept of absurdism in his essays, novels, and plays.

One of the central tenets of absurdism is the idea that life has no inherent meaning or purpose. According to this view, humans are born into a world that is fundamentally chaotic and meaningless, and our attempts to find meaning or purpose are ultimately futile. This can lead to feelings of despair and anxiety, as we struggle to come to terms with our own mortality and the apparent senselessness of our lives.

However, despite its bleak outlook, absurdism also offers a kind of freedom. If life has no inherent meaning or purpose, then we are free to create our own meaning and purpose. We are not bound by any pre-determined destiny or divine plan, but are free to shape our own lives according to our own values and desires.

One of the most famous expressions of absurdism is Camus' essay "The Myth of Sisyphus," in which he explores the story of Sisyphus, a figure from Greek mythology who was condemned to roll a boulder up a hill, only to watch it roll back down again for eternity. Camus sees Sisyphus as a metaphor for the human condition, in which we are constantly striving to achieve our goals, only to be thwarted by the futility and meaninglessness of our efforts.

Despite this, Camus argues that we must embrace the absurdity of our existence, and find a way to live with it. This means accepting that life is inherently meaningless, but also finding joy and fulfillment in the present moment, and in our connections with others. By embracing the absurdity of our lives, we can find a kind of freedom and authenticity that is otherwise impossible.

In conclusion, absurdism is a philosophical belief that emphasizes the irrationality and meaninglessness of human existence in a universe that is indifferent to our needs and desires. While it can lead to feelings of despair and anxiety, it also offers a kind of freedom, in which we are free to create our own meaning and purpose. Ultimately, the key to living with absurdism is to embrace it, and find joy and fulfillment in the present moment, and in our connections with others.
    """

    summary = summarize_text(text)
    print("Original text:")
    print(text)
    print("\nSummary:")
    print(summary)

if __name__ == "__main__":
    main()

