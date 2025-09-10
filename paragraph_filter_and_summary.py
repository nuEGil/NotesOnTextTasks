import hashlib
from FeatureCrafting import *
from transformers import pipeline, set_seed

def filter_unique_lines(text):
    # need to work on the echo filter
    lines = text.splitlines()
    # print(lines)
   
    seen = set()
    result = []

    for line in lines:
        # Hash the line (you can also just use the line itself if it's not huge)
        h = hashlib.sha256(line.encode("utf-8")).hexdigest()
        
        if h in seen:
            # Stop processing once a duplicate is found
            break
        else:
            seen.add(h)
            result.append(line)
    
    return '\n'.join(result)

if __name__ =='__main__':
    # xargs = get_args()
    # Book = '/mnt/f/ebooks_public_domain/crime and punishment.txt'
    # book_dat = FirstRead(Book)
    set_seed(42) # reproducablility

    paragraph_data_name = '/mnt/f/ebooks_public_domain/crime and punishment_pargraph_feats.csv'
    paragraph_data = pd.read_csv(paragraph_data_name)
    
    rms_mean = np.sqrt(np.mean(paragraph_data['avg_special_char'].to_numpy()**2))
    
    sub_data = paragraph_data[paragraph_data['avg_special_char'] > 5*rms_mean]
    print('sub data shape ', sub_data.shape)

    # creating the prompt
    paragraph_id = 5
    ptext = f'paragraph # {paragraph_id}: {sub_data['paragraph'].iloc[paragraph_id]}'
    task = """
    ---
    Given the paragraph above perform the following tasks
    1. Who are the people involved in this scene?
    2. What are the people in this paragraph doing?
    3. Where is this scene?
    4. When is this scene taking place?
    5. Why?
    ---
    Response: 
    """
    prompt = f'{ptext}\n{task}'

    # load the model.     
    pipe = pipeline('text-generation', model ='google/gemma-3-270m')
    
    # messages = [{"role":"user", "content":prompt}]
    outputs = pipe(prompt, max_new_tokens=256, num_return_sequences=1)    
    print(outputs[0]["generated_text"])
    # filtered_output = filter_unique_lines(outputs[0]["generated_text"])
    # print(filtered_output)
   