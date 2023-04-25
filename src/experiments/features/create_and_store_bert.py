from load_and_store_emb_batches import get_dfs, make_last_3_bert

if __name__ == "__main__":
    
    dfs = get_dfs()
    
    make_last_3_bert(dfs)