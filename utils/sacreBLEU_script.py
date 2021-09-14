
from sacrebleu import corpus_bleu, TOKENIZERS

SMOOTH_VARIANTS = ["exp", "floor", "none"]

    #%%
def compute_sacre_bleu(hypotheses, references,
                 smooth_method = "exp",
                 smooth_value = 0.0,
                 force = False,
                 lowercase = False,
                 tokenize = "none",
                 use_effective_order = False):

    hyp_joined = [" ".join(hyp) for hyp in hypotheses]
    ref_joined = [" ".join(ref) for ref in references]

    bleu = corpus_bleu(hyp_joined, [ref_joined],
                       smooth_method=smooth_method,
                       smooth_value=smooth_value,
                       force=force,
                       lowercase=lowercase,
                       tokenize=tokenize,
                       use_effective_order=use_effective_order)

    return bleu.score

