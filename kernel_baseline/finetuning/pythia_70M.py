from transformers import GPTNeoXForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer


#Load Pythia and tokenizer
model_name = "EleutherAI/pythia-70m-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#Tokenize
def tokenize(batch):
    return tokenizer(batch["text"], padding = "max_length", truncation = "True", max_length = 128)

tokenized = dataset.map(tokenize, batched = True)


#Training loop 
training_args = TrainingArguments(
    output_dir = "./pythia70m-sst2-fullft",
    per_device_train_batch_size = 16,
)