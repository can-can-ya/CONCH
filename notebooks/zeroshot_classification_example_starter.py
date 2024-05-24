from CONCH.conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer
import torch
from PIL import Image

# show all jupyter output
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Load model from checkpoint

model_cfg = 'conch_ViT-B-16'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
checkpoint_path = '/home/gjx/can_pretrained-model/conch/pytorch_model.bin'
model, preprocess = create_model_from_pretrained(model_cfg, checkpoint_path, device=device)
model.eval()

# Open an image and preprocess it

image = Image.open('../docs/roi1.jpg')
image_tensor = preprocess(image).unsqueeze(0).to(device)
image.resize((224, 224))

# Load tokenizer and specify some prompts. Simplicity we just use one prompt per class (lung adenocarcinoma vs. lung squamous cell carcinoma) here instead ensembling multiple prompts / prompt templates.

tokenizer = get_tokenizer()
classes = ['invasive ductal carcinoma',
           'invasive lobular carcinoma']
prompts = ['an H&E image of invasive ductal carcinoma',
           'an H&E image of invasive lobular carcinoma']

tokenized_prompts = tokenize(texts=prompts, tokenizer=tokenizer).to(device)

with torch.inference_mode():
    image_embedings = model.encode_image(image_tensor)
    text_embedings = model.encode_text(tokenized_prompts)
    sim_scores = (image_embedings @ text_embedings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()

print("Predicted class:", classes[sim_scores.argmax()])
print("Normalized similarity scores:", [f"{cls}: {score:.3f}" for cls, score in zip(classes, sim_scores[0])])