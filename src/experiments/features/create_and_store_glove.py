from load_and_store_emb_batches import get_dfs, create_and_store_all_embs_of_type

if __name__ == "__main__":
    
    dfs = get_dfs()

    create_and_store_all_embs_of_type(dfs, "glove")