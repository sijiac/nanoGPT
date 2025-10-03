# train a miniature character-level shakespeare model
# good for debugging and playing on macbooks and such

out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

import time
ENABLE_NANO_MLP = os.environ.get("NANO_MLP", "0") == "1"
ENABLE_NANO_ATTN = os.environ.get("NANO_ATTN", "0") == "1"
ENABLE_PE_ZERO_INIT = os.environ.get("ENABLE_PE_ZERO_INIT", "0") == "1"

timestamp = time.strftime('%Y%m%d.%H%M')

wandb_log = True # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = f'nanoGPT-{timestamp}-NANO_MLP={ENABLE_NANO_MLP}-NANO_ATTN={ENABLE_NANO_ATTN}-PE_ZERO={ENABLE_PE_ZERO_INIT}'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256 # context of up to 256 previous characters

# baby GPT model :)
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3 # with baby networks can afford to go a bit higher
max_iters = 5000
lr_decay_iters = 5000 # make equal to max_iters usually
min_lr = 1e-4 # learning_rate / 10 usually
beta2 = 0.99 # make a bit bigger because number of tokens per iter is small

warmup_iters = 100 # not super necessary potentially

# on macbook also add
# device = 'cpu'  # run on cpu only
# compile = False # do not torch compile the model
