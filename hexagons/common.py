from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM

MODEL_CLASSES = {
    "seq2seq": AutoModelForSeq2SeqLM,
    "causal": AutoModelForCausalLM
}
