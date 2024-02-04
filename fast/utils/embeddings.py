import numpy as np

def add_embeddings(embeddings, column_ids, embedding_size, is_UV, is_diff, is_mult, use_abs):
    '''
    Add embeddings at specific column_ids. For example, for the matrix
    [Z1, Z2, U1, V1, Z3, U2, V2], if we want to replace this with:
    [Z1, Z2, U1 - V1, Z3, U2, V2, U2 - V2, U2 * V2], we provide the parameters:
    is_UV = [False, True] : U1, V1 are NOT kept, but U2, V2 are kept
    is_diff = [True, True] : both U1 - V1 and U2 - V2 are added
    is_mult = [False, True] : U1 * V1 is not included, U2 * V2 is included

    Args:
        embeddings : original matrix to replace
        column_ids : location of Ux (we assume Vx immedeately follows Ux), 
                     for the above example, we would provide column_ids = [2, 5].
                     If you DO NOT want to replace a certain Ux, simply don't include its id in column_ids
        is_UV : should Ux, Vx be included
        is_diff : should Ux - Vx be included
        is_mult : should Ux * Vx be included
        is_abs  : should Ux - Vx be absolute valued
    '''

    id_delta = 0 # keep track of changes to embedding inserts/deletions
    for id, column_id in enumerate(column_ids):

        if id>0:
            if not is_UV[id-1]:
                id_delta -= 2
            if is_diff[id-1]:
                id_delta += 1
            if is_mult[id-1]:
                id_delta += 1

        start_id = (column_id + id_delta) * embedding_size
        U_id = start_id + embedding_size
        V_id = start_id + (2*embedding_size)

        U = embeddings[:, start_id:U_id]
        V = embeddings[:, U_id:V_id]

        if is_diff[id] and not is_mult[id]:
            new_embeddings = np.abs(U - V) if use_abs else (U-V)
        elif not is_diff[id] and is_mult[id]:
            new_embeddings = U * V
        else: # both
            subtracted = np.abs(U - V) if use_abs else (U-V)
            new_embeddings = np.hstack([subtracted, (U * V)])

        if is_UV[id]:                  
            embeddings = np.hstack([
                embeddings[:, :V_id],       # Part of original matrix before replacement
                new_embeddings,             # New embeddings to insert
                embeddings[:, V_id:]        # Part of original matrix after replacement
            ])
        else:
            embeddings = np.hstack([
                embeddings[:, :start_id],   # Part of original matrix before replacement
                new_embeddings,             # New embeddings to insert
                embeddings[:, V_id:]        # Part of original matrix after replacement
            ])
    return embeddings