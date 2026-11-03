#Observons comment regex fonctionne.
import regex

GPT4_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

phrases = [
    "L'été, j'ai vu 12345 astéroïdes\xa0!",
    "Qu’il vienne « demain » à 18h30.",
    "def f(x):\n    return x",
]

if __name__ == "__main__":
    for phrase in phrases:
        print(regex.findall(GPT4_PATTERN, phrase))