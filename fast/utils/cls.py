# Define a function to extract CLS embeddings from input text
def extract_cls_embeddings(model, text_array):
    '''
    Get CLS embeddings for each text item in the array.
    Args:
    - text_array (list): List of input text.
    Returns:
    - NumPy array (len(text_array), 768): CLS embeddings (768,) for each text.
    '''
    # Tokenize sentences
    encoded_input = tokenizer(text_array, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input) 
    # Model_output is a dictionary with 'last_hidden_state' key that stores final hidden states (all token embeddings)
    last_hidden_state = model_output['last_hidden_state'] # Same as model_output[0]
   
    # Extract the CLS embedding (index 0) from the last hidden state (3D tensor)
    cls_embedding = last_hidden_state[:, 0, :]
    # keep all data in batch, keep all hidden state features, but select first element from hidden states (hidden state corresponding to [CLS] token)

    # Convert to numpy array
    cls_embedding = cls_embedding.numpy()

    return cls_embedding

# Example
def cls_embedding_example(model):
    examples = ["sentjnmnsd.", "!!kskfjka@"]
    cls_embeddings_array = extract_cls_embeddings(model, examples)
    print(cls_embeddings_array)