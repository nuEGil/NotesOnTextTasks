from transformers import pipeline, set_seed

'''
This version will run locally. 
'''

if __name__ == '__main__':
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)

    prompt_ = 'Hello, my name is: '
    gpt2_outputs = generator(prompt_, max_new_tokens = 128, num_return_sequences=3)

    for i in range(3):
        print(gpt2_outputs[i]['generated_text'])