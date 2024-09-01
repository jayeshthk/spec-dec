"""You should implement both of these methods.

Vanilla edits should just be a custom generate loop with a huggingface
transformer.

Speculative edits should implement the speculative editing algorithm.

To test these, make sure they work on the prompt provided in the README"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
def speculative_edit(prompt: str, draft: str, max_tokens: int) -> str:

    prompt_inputs = tokenizer(prompt, return_tensors='pt')
    draft_inputs = tokenizer(draft, return_tensors='pt')

    speculative_tokens = draft_inputs['input_ids'][0].clone().detach()

    prompt_length = len(prompt_inputs['input_ids'][0])

    for idx in range(len(speculative_tokens)):
        with torch.no_grad():
            outputs = model(prompt_inputs['input_ids'])[0]
        
        predicted_token_id = torch.argmax(outputs[:, idx, :], dim=-1).item()
        
        if predicted_token_id != speculative_tokens[idx].item():
            generated_output = model.generate(
                prompt_inputs['input_ids'][:, :idx+1],
                max_length=prompt_length + max_tokens,
                do_sample=False,
                temperature=0.0
            )
            
            generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
            return generated_text

    return tokenizer.decode(speculative_tokens, skip_special_tokens=True)
    # raise NotImplementedError("This function is not implemented yet.")

def vanilla_edit(prompt: str, max_tokens: str) -> str:
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=inputs['input_ids'].shape[1] + max_tokens,
        do_sample=False,  # Greedy decoding
        temperature=0.0
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text
    # raise NotImplementedError("This function is not implemented yet.")
prompt = """
````txt
Please add a single comment in code at return statement

```py
def return_tensors(input:str):
    return torch.randn((2,4))
```

```py
````
"""


# Testing vanilla edit
vanilla_output = vanilla_edit(prompt, max_tokens=100)
print("Vanilla Edit Output: \n", vanilla_output)

# generated output: 
# """
# Vanilla Edit Output: 
 
# ````txt
# Please add a single comment in code at return statement

# ```py
# def return_tensors(input:str):
#     return torch.randn((2,4))
# ```

# ```py
# ````

# ## Answer

# ```py
# def return_tensors(input:str):
#     # This function returns a tensor of size (2,4)
#     return torch.randn((2,4))
# ```

# ## Explanation

# The comment added at the return statement provides a brief explanation of what the function does. This is useful for both the programmer who might be reading the code and for any automated documentation generation tools.

# ##
# Selection deleted



# """


draft="""
Review the code below and add a comment on the return statement if it is missing:

```py
def return_tensors(input: str):
    return torch.randn((2, 4))
```
"""

# Testing speculative edit
speculative_output = speculative_edit(prompt, draft, max_tokens=100)
print("Speculative Edit Output: \n", speculative_output)

# generated output (unsuccessful): 
# """
# Speculative Edit Output: 
 
#     public function getId(): ?int
#     {
#         return $this->id;
#     }

#     public function getName(): ?string
#     {
#         return $this->name;
#     }

#     public function setName(string $name): self
#     {
#         $this->name = $name;

#         return $this;
#     }

#     public function getDescription(): ?string
#     {
#         return $this->description;
#     }

#     public function setDescription(string $description): self
#     {
#         $this->description = $description;

#         return $this;
#     }

#     public function getPrice(): ?float
#     """