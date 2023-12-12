import torch
import torch.nn.functional as F

# Define a function for mean pooling of tokens from input text
def mean_pooling(model_output, attention_mask):
    '''
    Get mean of token embeddings for all text items in the array (excludes padding tokens).
    Args:
    - model_output (dictionary): Outputs of model, containing last_hidden_state, pooler_output, hidden_states, attentions.
    - attention_mask (tensor): Mask that indicates which tokens are actual tokens from the input (1s) and which ones are padding tokens (0s).
    Returns:
    - PyTorch tensor (len(text_array), 768): mean-pooled embeddings (768,) for each text.
    '''
    # Extract token embeddings from the model_output -- first element contains embeddings for each token in the input sequence.
    token_embeddings = model_output[0]

    # Adjust mask dimensions -- unsqueeze to add a dimension to attention_mask, expand to match dimensions of token_embeddings.
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

    # Apply mask to token embeddings
    masked_embeddings = token_embeddings * input_mask_expanded

    # Compute mean -- sum along token dimension axis=1 and divide by sum of mask values (i.e. the count of non-padding tokens), avoiding division by 0
    mean_embeddings = torch.sum(masked_embeddings, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
    # Normalize
    mean_embeddings = F.normalize(mean_embeddings, p=2, dim=1)
    
    return mean_embeddings 