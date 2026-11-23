from tokenizer.tokenizer import decode

def make_chunks(tokens, chunk_size: int, overlap: int, source: str, min_len: int, id_start: int = 0) -> list:
    
    chunks = []
    merges = 0
    i = 0 

    while ((i+1) * (chunk_size - overlap) + chunk_size <= len(tokens)) :
        ids = tokens[i * (chunk_size - overlap): i * (chunk_size - overlap) + chunk_size]
        chunk = {
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids)
        }
        chunks.append(chunk)
        i+=1
    if (len(tokens) - (i+1) * (chunk_size - overlap) < min_len):
        ids = tokens[i * (chunk_size - overlap):]
        chunks.append({
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids)
        })
        merges += 1
    else:
        ids = tokens[i * (chunk_size - overlap):i * (chunk_size - overlap) + chunk_size]
        chunks.append({
                    "id": id_start + len(chunks),
                    "source": source,
                    "token_ids": ids,
                    "text": decode(ids)
                })
        chunks.append({
            "id": id_start + len(chunks),
            "source": source,
            "token_ids": ids,
            "text": decode(ids)
        }) 
    return chunks, {"pass": len(chunks), "tokens": len(tokens), "merges": merges}


    
