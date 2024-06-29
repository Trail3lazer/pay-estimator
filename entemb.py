from keras import layers, losses, optimizers
from keras import Model
from keras import saving
import os
import settings
from typing import Dict

def get_or_create_entity_embeddings(
    df: dict[str, list] = None, 
    cat_cols: list[str] = [], 
    cont_cols: list[str] = [], 
    min_embs: int = 50
    ):
    
    if(os.path.isfile(settings.ENTEMB_MODEL)):
        print("Retrieving an existing entity embedding model from "+settings.ENTEMB_MODEL)
        model = saving.load_model(settings.ENTEMB_MODEL)
    else: 
        model = create_entity_embeddings(df, cat_cols, cont_cols, min_embs=50)
        model.save(settings.ENTEMB_MODEL)
    return model

def create_entity_embeddings(
    df: dict[str, list] = None, 
    cat_cols: list[str] = [], 
    cont_cols: list[str] = [], 
    min_embs: int = 50
    ):
    
    cats_len = {col: df[col].nunique() for col in cat_cols}
    cats_emb = {col: min(min_embs,(cats_len[col]//2+1)) for col in cat_cols}
    
    inputs = []
    concat = []
    
    for cat in cat_cols:
        input = layers.Input(shape=(1,), name=cat)
        inputs.append(input)
        embedding = layers.Embedding((cats_len[cat]+1), cats_emb[cat], name='embeddings')(input)
        embedding = layers.Reshape((cats_emb[cat],))(embedding)
        concat.append(embedding)
        
    for cont in cont_cols:
        input = layers.Input(shape=(1,), name=cont)
        inputs.append(input)
        concat.append(input)
        
    lays = layers.Concatenate()(concat)
    lays = layers.Dense(100, activation= 'relu')(lays)
    lays = layers.Dense(1)(lays)
    
    model = Model(inputs, lays)
    model.compile(loss=losses.MSLE, optimizer=optimizers.Adam(), metrics=[losses.MSLE])
    
    return model
