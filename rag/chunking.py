from tokenizer.tokenizer import decode

def make_chunks(tokens: list, chunk_size: int, overlap: int, source: str, min_len: int) -> list:
    chunks = []
    i = 0 
    while ((i+1) * (chunk_size - overlap) + chunk_size <= len(tokens)) :
        chunk = {
            "id": i,
            "source": source,
            "token_ids": tokens[i * (chunk_size - overlap): i * (chunk_size - overlap) + chunk_size],
            "text": decode(chunk["token_ids"])
        }
        chunks.append(chunk)
        i+=1
    if (len(tokens) - (i+1) * (chunk_size - overlap) < min_len):
        chunks.append({"id": i, "source": source, "token_ids": tokens[i * (chunk_size - overlap):], "text": decode(chunk["token_ids"])})
    else:
        chunks.append({"id": i, "source": source, "token_ids": tokens[i * (chunk_size - overlap):i * (chunk_size - overlap) + chunk_size], "text": decode(chunk["token_ids"])})
        chunks.append({"id": i+1, "source": source, "token_ids": tokens[(i+1) * (chunk_size - overlap):], "text": decode(chunk["token_ids"])})
    return chunks


    
