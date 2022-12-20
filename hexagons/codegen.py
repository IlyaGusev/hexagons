from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Salesforce/codegen-6B-multi"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_side="left")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).cuda()

def codegen_predict(prompt):
    input_ids = tokenizer(
        prompt,
        truncation=True,
        return_tensors="pt",
        max_length=1024
    ).input_ids.cuda()
    bad_words = ["#"]
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids
    generated_ids = model.generate(
        input_ids,
        max_new_tokens=512,
        bad_words_ids=bad_words_ids,
        num_beams=5
    )
    result = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return result
