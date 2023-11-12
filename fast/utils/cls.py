# Define a function to extract CLS embeddings from input text
def extract_cls_embeddings(model_output):
    '''
    Get CLS embeddings for each text item in the array.
    Args:
    - model_outputs (dictionary): Outputs of model, containing last_hidden_state, pooler_output, hidden_states, attentions.
    Returns:
    - NumPy array (len(text_array), 768): CLS embeddings (768,) for each text.
    '''
    # Model_output is a dictionary with 'last_hidden_state' key that stores final hidden states (all token embeddings)
    # Same as model_output[0]
    last_hidden_state = model_output['last_hidden_state']

    # Extract the CLS embedding (index 0) from the last hidden state (3D tensor)
    cls_embedding = last_hidden_state[:, 0, :]
    # keep all data in batch, keep all hidden state features, but select first element from hidden states (hidden state corresponding to [CLS] token)

    # Convert to numpy array
    cls_embedding = cls_embedding.numpy()

    return cls_embedding
