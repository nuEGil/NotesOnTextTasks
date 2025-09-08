from transformers import pipeline, set_seed

# grab context
def GetTextFile(x):
    with open(x, "r", encoding = "utf-8") as f:
        return f.read()
    
# grab just the model output
def generate_continuation(generator, prompt, max_length=256):
    output = generator(prompt, max_length=max_length, do_sample=True)[0]["generated_text"]
    # Strip off the original prompt from the start
    return output[len(prompt):].strip()

if __name__ == '__main__':    
    set_seed(42) # reproducablility
    # grab relavent text to work with 
    context = GetTextFile('local_context.txt') # context 
    input_ = 'How should you proceed with treatment?'
    prompt = f'{context}\n{input_}'

    print('Input: ', prompt)
    # generator = pipeline('text-generation', model='gpt2')
    # you need an access token for some models like gemma
    generator = pipeline('text-generation', model ='google/gemma-3-270m')
    output_text = generate_continuation(generator, prompt, max_length=256)
    print(output_text)
    