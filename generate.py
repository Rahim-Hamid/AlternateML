from transformers import pipeline

gen = pipeline(
    "text-generation",
    model = "gpt2",
    do_sample = True,
    temperature = 0.9,
    max_new_tokens = 10
)

def gen_title(similar_titles):
    prompt = "Generate a short, poetic artwork title inspired by the following titles:\n"
    for t in similar_titles:
        prompt += f"- {t}\n"
    prompt += "Title:"

    out = gen(
        prompt,
        max_new_tokens=6,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
        eos_token_id=gen.tokenizer.eos_token_id
    )[0]["generated_text"]

    return out.split("Title:")[-1].split("\n")[0].strip()