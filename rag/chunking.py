from tokenizer.tokenizer import decode


def make_chunks(tokens, vocab, chunk_size: int, overlap: int, source: str, min_len: int, id_start: int = 0) -> list:

    if not 0 <= overlap < chunk_size:
        raise ValueError("Découpage impossible.")
    if not tokens:
        return [], {"passages": 0, "tokens": 0, "fusions": 0}
    
    chunks = []
    merges = 0
    i = 0 

    while ((i+1) * (chunk_size - overlap) + chunk_size <= len(tokens)) :
        ids = tokens[i * (chunk_size - overlap): i * (chunk_size - overlap) + chunk_size]
        chunk = {
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids, vocab)
        }
        chunks.append(chunk)
        i+=1
    if (len(tokens) - (i+1) * (chunk_size - overlap) < min_len):
        merge = bool(chunks)
        ids = tokens[i * (chunk_size - overlap):]
        chunks.append({
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids, vocab)
        })
        merges += merge
    else:
        ids = tokens[i * (chunk_size - overlap):i * (chunk_size - overlap) + chunk_size]
        chunks.append({
                    "id": id_start + len(chunks),
                    "source": source,
                    "token_ids": ids,
                    "text": decode(ids, vocab)
                })
        ids = tokens[(i+1) * (chunk_size - overlap):]
        chunks.append({
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids, vocab)
        }) 
    return chunks, {"passages": len(chunks), "tokens": len(tokens), "fusions": merges}


    
