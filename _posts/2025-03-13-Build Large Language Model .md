---
title: 'Build Large Language Model'
date: 2025-03-13
permalink: /posts/2025/3/Build Large Language Model/
tags:
  - python
  - pytorch
  - attention
  - LLM
---



# 阅读图书Build Large Language Model



日期: 2025年2月9日 → 2025年2月28日
状态: 进行中

AI 安全

注意力机制

[https://blog.csdn.net/weixin_42110638/article/details/134011134](https://blog.csdn.net/weixin_42110638/article/details/134011134)

深度解析注意力机制

[https://mp.weixin.qq.com/s/Qlf33S3UkxO8Kui1XfH_Fg](https://mp.weixin.qq.com/s/Qlf33S3UkxO8Kui1XfH_Fg)

蒸馏算法，使用大模型训练出小模型，大模型在给小模型训练时候会给出正确数据的同时会给出极小概率的其他可能性，比如在识别手写2图片时候，在告诉这个是2的同时会给他0.00001的可能性为3，0.00000001可能性为7，在小模型没有遇到过3，7的情况下也有可能识别出来这个是3，7.这个就可以提高模型泛化，同时由大模型训练出的小模型比单独训练的小模型准确率要好。

对这个方向研究在IDS上的运用。尝试研究。

# 开始

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image.png)

# 第2章 Working with Text Data

介绍从text数据转化为token_id的过程，介绍原理。可以直接使用

```python
import tiktoken 
 
tokenizer = tiktoken.get_encoding("gpt2")
```

1. **文本到数值向量的转换**：LLMs无法直接处理原始文本，因此需要将文本转换为数值向量（嵌入）。嵌入将离散数据（如单词或图像）转换为连续的向量空间，使其适用于神经网络操作。
2. **分词与标记化**：首先，原始文本被分解为标记（tokens），标记可以是单词或字符。然后，这些标记被转换为整数表示，称为标记ID。
3. **特殊标记**：为了增强模型的理解能力，可以添加特殊标记（如`<|unk|>`表示未知单词，`<|endoftext|>`表示文本结束），以处理不同上下文。
4. **字节对编码（BPE）**：GPT-2和GPT-3等模型使用BPE分词器，能够通过将未知单词分解为子词单元或单个字符来高效处理它们。
5. **滑动窗口方法**：在训练LLMs时，使用滑动窗口方法在标记化数据上生成输入-目标对。
6. **嵌入层**：在PyTorch中，嵌入层通过查找操作检索与标记ID对应的向量，生成连续的标记表示，这对训练深度学习模型至关重要。
7. **位置嵌入**：为了表示标记在序列中的位置，有两种主要的位置嵌入方法：绝对位置嵌入和相对位置嵌入。OpenAI的GPT模型使用绝对位置嵌入，将其添加到标记嵌入向量中，并在模型训练过程中进行优化。

# 第3章 Coding Attention Mechanisms

主要介绍了注意力机制及其在大型语言模型（LLMs）中的应用

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image 1.png)

1. **注意力机制的作用**：注意力机制将输入元素转换为增强的上下文向量表示，这些表示包含了所有输入的信息。
2. **自注意力机制**：自注意力机制通过计算输入元素的加权和来生成上下文向量表示。在简化的注意力机制中，注意力权重通过点积计算。
3. **点积与矩阵乘法**：点积是对两个向量逐元素相乘后求和，矩阵乘法可以高效地替代嵌套循环，使计算更紧凑和高效。
4. **缩放点积注意力**：LLMs中使用的自注意力机制（称为缩放点积注意力）引入了可训练的权重矩阵，用于计算输入的中间变换：查询（queries）、值（values）和键（keys）。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%202.png)
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%203.png)
    
5. **因果注意力掩码(causal attention mask)**：在从左到右生成文本的LLMs中，使用因果注意力掩码来防止模型访问未来的标记（tokens）。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%204.png)
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%205.png)
    
6. **Dropout掩码**：除了因果注意力掩码，还可以添加Dropout掩码以减少LLMs的过拟合。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%206.png)
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%207.png)
    
7. **多头注意力**：基于Transformer的LLMs使用多头注意力机制，即多个因果注意力模块的堆叠。通过批处理矩阵乘法可以更高效地实现多头注意力模块。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%208.png)
    

# 第4章Implementing a GPT model from  Scratch To Generate Text

说明GPT模型的核心组件（如层归一化、快捷连接和Transformer块）、模型规模以及文本生成的基本原理，同时强调了模型训练的关键作用

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%209.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2010.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2011.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2012.png)

1. **层归一化（Layer Normalization）**：通过确保每一层的输出具有一致的均值和方差，层归一化能够稳定训练过程。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2013.png)
    
    ```python
    class LayerNorm(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.eps = 1e-5
            self.scale = nn.Parameter(torch.ones(emb_dim))
            self.shift = nn.Parameter(torch.zeros(emb_dim))
    
        def forward(self, x):
            mean = x.mean(dim=-1, keepdim=True)
            var = x.var(dim=-1, keepdim=True, unbiased=False)
            norm_x = (x - mean) / torch.sqrt(var + self.eps)
            return self.scale * norm_x + self.shift
    ```
    
2. **forward network**
   
    使用GELU activations而不是ReLU activations防止梯度消失
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2014.png)
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2015.png)
    
3. **快捷连接（Shortcut Connections）**：快捷连接通过将某一层的输出直接传递到更深的层，跳过一个或多个层，从而缓解深度神经网络（如LLMs）训练中的梯度消失问题。
    - 在深层网络中，梯度在反向传播时需要通过多个层逐层传递。如果每一层的梯度值较小，经过多层传递后，梯度可能会变得非常小，甚至趋近于零（梯度消失）。
    - 快捷连接通过跳过某些层，为梯度提供了一条**直接的传播路径**，使得梯度能够更高效地传递到浅层网络，避免因多层传递而导致的梯度衰减。
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2016.png)
    
    ```python
    class ExampleDeepNeuralNetwork(nn.Module):
        def __init__(self, layer_sizes, use_shortcut):
            super().__init__()
            self.use_shortcut = use_shortcut
            self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
                nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
            ])
    
        def forward(self, x):
            for layer in self.layers:
                # Compute the output of the current layer
                layer_output = layer(x)
                # Check if shortcut can be applied
                if self.use_shortcut and x.shape == layer_output.shape:
                    x = x + layer_output
                else:
                    x = layer_output
            return x
    
    def print_gradients(model, x):
        # Forward pass
        output = model(x)
        target = torch.tensor([[0.]])
    
        # Calculate loss based on how close the target
        # and output are
        loss = nn.MSELoss()
        loss = loss(output, target)
        
        # Backward pass to calculate the gradients
        loss.backward()
    
        for name, param in model.named_parameters():
            if 'weight' in name:
                # Print the mean absolute gradient of the weights
                print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
    ```
    
4. **Transformer块**：Transformer块是GPT模型的核心结构组件，结合了掩码多头注意力模块和全连接的前馈神经网络（使用GELU激活函数）。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2017.png)
    
    ```python
    
    class TransformerBlock(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                context_length=cfg["context_length"],
                num_heads=cfg["n_heads"], 
                dropout=cfg["drop_rate"],
                qkv_bias=cfg["qkv_bias"])
            self.ff = FeedForward(cfg)
            self.norm1 = LayerNorm(cfg["emb_dim"])
            self.norm2 = LayerNorm(cfg["emb_dim"])
            self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    
        def forward(self, x):
            # Shortcut connection for attention block
            shortcut = x
            x = self.norm1(x)
            x = self.att(x)  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back
    
            # Shortcut connection for feed forward block
            shortcut = x
            x = self.norm2(x)
            x = self.ff(x)
            x = self.drop_shortcut(x)
            x = x + shortcut  # Add the original input back
    
            return x
    ```
    
5. **GPT模型**：GPT模型是由多个重复的Transformer块组成的大型语言模型（LLMs），参数规模从数百万到数十亿不等。不同规模的GPT模型（如1.24亿、3.45亿、7.62亿和15.42亿参数）可以使用相同的Python类（如`GPTModel`）实现。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2018.png)
    
    ```python
    class GPTModel(nn.Module):
        def __init__(self, cfg):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
            self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
            self.drop_emb = nn.Dropout(cfg["drop_rate"])
            
            self.trf_blocks = nn.Sequential(
                *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
            
            self.final_norm = LayerNorm(cfg["emb_dim"])
            self.out_head = nn.Linear(
                cfg["emb_dim"], cfg["vocab_size"], bias=False
            )
    
        def forward(self, in_idx):
            batch_size, seq_len = in_idx.shape
            tok_embeds = self.tok_emb(in_idx)
            pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
            x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
            x = self.drop_emb(x)
            x = self.trf_blocks(x)
            x = self.final_norm(x)
            logits = self.out_head(x)
            return logits
    ```
    
6. **文本生成**：GPT模型的文本生成能力涉及将输出张量解码为人类可读的文本，基于给定的输入上下文逐词预测。未经训练的GPT模型会生成不连贯的文本，这凸显了模型训练对于生成连贯文本的重要性。
   
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2019.png)
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2020.png)
    
    ```python
    def generate_text_simple(model, idx, max_new_tokens, context_size):
        # idx is (batch, n_tokens) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # Crop current context if it exceeds the supported context size
            # E.g., if LLM supports only 5 tokens, and the context size is 10
            # then only the last 5 tokens are used as context
            idx_cond = idx[:, -context_size:]
            
            # Get the predictions
            with torch.no_grad():
                logits = model(idx_cond)
            
            # Focus only on the last time step
            # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
            logits = logits[:, -1, :]  
    
            # Apply softmax to get probabilities
            probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)
    
            # Get the idx of the vocab entry with the highest probability value
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
    
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)
    
        return idx
     
     start_context = "Hello, I am"
    
    encoded = tokenizer.encode(start_context)
    print("encoded:", encoded)
    
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("encoded_tensor.shape:", encoded_tensor.shape)
    model.eval() # disable dropout
    
    out = generate_text_simple(
        model=model,
        idx=encoded_tensor, 
        max_new_tokens=6, 
        context_size=GPT_CONFIG_124M["context_length"]
    )
    
    print("Output:", out)
    print("Output length:", len(out[0]))
    decoded_text = tokenizer.decode(out.squeeze(0).tolist())
    print(decoded_text)
    `Hello, I am Featureiman Byeswickattribute argue`
    ```
    
    ![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2021.png)
    

# 第5章 Pretraining on Unlabeled Data

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2022.png)

## Evaluating generative text models

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2023.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2024.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2025.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2026.png)

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2027.png)

```python
import os
import urllib.request

file_path = "the-verdict.txt"
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

if not os.path.exists(file_path):
    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode('utf-8')
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(text_data)
else:
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
from previous_chapters import create_dataloader_v1

# Train/validation ratio
train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]

torch.manual_seed(123)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)

val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M["context_length"],
    stride=GPT_CONFIG_124M["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)
# Sanity check

if total_tokens * (train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the training loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "increase the `training_ratio`")

if total_tokens * (1-train_ratio) < GPT_CONFIG_124M["context_length"]:
    print("Not enough tokens for the validation loader. "
          "Try to lower the `GPT_CONFIG_124M['context_length']` or "
          "decrease the `training_ratio`")

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Note:
# Uncommenting the following lines will allow the code to run on Apple Silicon chips, if applicable,
# which is approximately 2x faster than on an Apple CPU (as measured on an M3 MacBook Air).
# However, the resulting loss values may be slightly different.

#if torch.cuda.is_available():
#    device = torch.device("cuda")
#elif torch.backends.mps.is_available():
#    device = torch.device("mps")
#else:
#    device = torch.device("cpu")
#
# print(f"Using {device} device.")

model.to(device) # no assignment model = model.to(device) necessary for nn.Module classes

torch.manual_seed(123) # For reproducibility due to the shuffling in the data loader

with torch.no_grad(): # Disable gradient tracking for efficiency because we are not training, yet
    train_loss = calc_loss_loader(train_loader, model, device)
    val_loss = calc_loss_loader(val_loader, model, device)

print("Training loss:", train_loss)
print("Validation loss:", val_loss)
```

## 训练模型

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2028.png)

```python
def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

# Note:
# Uncomment the following code to calculate the execution time
# import time
# start_time = time.time()

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

num_epochs = 10
train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)

# Note:
# Uncomment the following code to show the execution time
# end_time = time.time()
# execution_time_minutes = (end_time - start_time) / 60
# print(f"Training completed in {execution_time_minutes:.2f} minutes.")
```

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2029.png)

## **Temperature Scaling**

**Temperature Scaling（温度缩放）** 是一种用于校准深度学习模型（尤其是分类模型）输出概率的技术。它通常用于提高模型预测概率的可靠性，使其更接近真实概率分布。温度缩放是 **模型校准（Model Calibration）** 的一种简单而有效的方法。

### 1. **背景：模型校准问题**

在分类任务中，深度学习模型通常会输出每个类别的概率（通过 softmax 函数）。然而，这些概率并不总是准确的，尤其是当模型过于自信或不够自信时：

- **过度自信**：模型输出的概率值过高（例如，预测某个类别的概率为 0.99，但实际上并不准确）。
- **不够自信**：模型输出的概率值过低（例如，预测某个类别的概率为 0.6，但实际上应该更高）。

模型校准的目标是调整模型的输出概率，使其更接近真实概率分布。

---

### 2. **温度缩放的原理**

温度缩放通过在 softmax 函数中引入一个 **温度参数 \( T \)** 来调整模型的输出概率。具体来说，softmax 函数的公式被修改为：

\[
\text{softmax}(z_i) = \frac{e^{z_i / T}}{\sum_{j=1}^N e^{z_j / T}}
\]

其中：

- \( z_i \) 是模型对第 \( i \) 个类别的 logit（未归一化的预测值）。
- \( T \) 是温度参数。
- \( N \) 是类别的总数。

### 温度参数 \( T \) 的作用：

- **\( T = 1 \)**：这是标准的 softmax 函数，不进行任何调整。
- **\( T > 1 \)**：增大温度会使得输出概率分布更加平滑，降低模型的置信度（概率值更接近均匀分布）。
- **\( T < 1 \)**：减小温度会使得输出概率分布更加尖锐，增加模型的置信度（概率值更接近 0 或 1）。

---

### 3. **温度缩放的实现**

温度缩放的实现非常简单，通常包括以下步骤：

1. 在验证集上训练一个温度参数 \( T \)。
2. 将训练好的 \( T \) 应用于测试集或实际推理中，调整模型的输出概率。

### 代码示例：

```python
import torch
import torch.nn.functional as F

# 假设模型的 logits 输出
logits = torch.tensor([[2.0, 1.0, 0.1]])

# 标准 softmax（T=1）
probs = F.softmax(logits, dim=-1)
print("Standard softmax:", probs)  # 输出: tensor([[0.6590, 0.2424, 0.0986]])

# 温度缩放（T=2）
T = 2
scaled_probs = F.softmax(logits / T, dim=-1)
print("Temperature scaled (T=2):", scaled_probs)  # 输出: tensor([[0.5423, 0.3380, 0.1197]])

# 温度缩放（T=0.5）
T = 0.5
scaled_probs = F.softmax(logits / T, dim=-1)
print("Temperature scaled (T=0.5):", scaled_probs)  # 输出: tensor([[0.7489, 0.2100, 0.0411]])

```

### 输出结果：

- 当 \( T = 2 \) 时，概率分布更加平滑，模型的置信度降低。
- 当 \( T = 0.5 \) 时，概率分布更加尖锐，模型的置信度增加。

---

### 4. **如何选择温度参数 \( T \)**

温度参数 \( T \) 通常通过在验证集上优化来获得。具体步骤如下：

1. 在验证集上计算模型的 logits 和真实标签。
2. 使用优化方法（如梯度下降）最小化负对数似然损失（Negative Log-Likelihood, NLL），找到最佳的 \( T \)。

### 代码示例：

```python
# 假设验证集的 logits 和标签
val_logits = torch.tensor([[2.0, 1.0, 0.1], [1.0, 2.0, 0.1]])
val_labels = torch.tensor([0, 1])  # 真实标签

# 定义温度参数 T（初始值为 1.0）
T = torch.tensor(1.0, requires_grad=True)

# 优化器
optimizer = torch.optim.LBFGS([T], lr=0.01)

# 优化过程
def eval():
    optimizer.zero_grad()
    loss = F.cross_entropy(val_logits / T, val_labels)
    loss.backward()
    return loss

optimizer.step(eval)

print("Optimized T:", T.item())  # 输出优化后的温度参数

```

---

### 5. **温度缩放的应用场景**

- **模型校准**：提高模型输出概率的可靠性，使其更接近真实概率分布。
- **不确定性估计**：在需要模型输出不确定性时（如医疗诊断、自动驾驶等），温度缩放可以帮助更好地量化模型的不确定性。
- **集成方法**：在模型集成（Ensemble）中，温度缩放可以用于调整每个子模型的输出概率。

---

### 6. **温度缩放的优缺点**

### 优点：

- 简单易实现，只需一个额外的参数 \( T \)。
- 计算开销小，适用于大规模模型。
- 可以有效提高模型的校准性能。

### 缺点：

- 只能调整概率分布的平滑度，无法改变模型的排序能力（即模型的预测顺序不变）。
- 对于某些复杂任务，可能需要更复杂的校准方法。

---

### 总结

- **温度缩放** 是一种简单而有效的模型校准技术，通过调整 softmax 函数中的温度参数 \( T \) 来优化模型的输出概率。
- 它广泛应用于分类任务中，尤其是在需要可靠概率估计的场景（如医疗、金融等）。
- 温度缩放的实现非常简单，且计算开销小，是模型校准的首选方法之一。

如果你有更多问题，欢迎继续讨论！

![阅读图书Build Large Language Model\image.png](阅读图书Build Large Language Model\image%2030.png)

## **Top-k Sampling**

**Top-k Sampling（Top-k 采样）** 是一种用于生成文本的采样策略，通常用于语言模型（如 GPT 等）的文本生成任务中。它的核心思想是从模型预测的概率分布中，选择概率最高的前 \( k \) 个词（或标记，token），然后从这 \( k \) 个词中进行采样，而不是从整个词汇表中采样。

---

### 1. **背景：文本生成中的采样问题**

在文本生成任务中，语言模型会输出一个概率分布，表示每个词（或标记）作为下一个词的可能性。传统的采样方法（如贪心搜索或随机采样）存在以下问题：

- **贪心搜索（Greedy Search）**：总是选择概率最高的词，容易导致生成的文本过于单调和重复。
- **随机采样（Random Sampling）**：从整个词汇表中随机采样，可能会导致生成不连贯或不符合语境的文本。

Top-k 采样是一种折衷方法，既避免了贪心搜索的单调性，又减少了随机采样的不确定性。

---

### 2. **Top-k 采样的原理**

Top-k 采样的核心思想是：

1. 从模型输出的概率分布中，选择概率最高的前 \( k \) 个词。
2. 对这 \( k \) 个词的概率重新归一化（使其和为 1）。
3. 从这 \( k \) 个词中随机采样一个词作为下一个词。

### 数学公式：

假设模型输出的概率分布为 \( P(x) \)，Top-k 采样的步骤如下：

1. 选择概率最高的前 \( k \) 个词，记为 \( V_{\text{top-k}} \)。
2. 重新归一化概率：
\[
P_{\text{top-k}}(x) = \begin{cases}
\frac{P(x)}{\sum_{x' \in V_{\text{top-k}}} P(x')} & \text{if } x \in V_{\text{top-k}} \\
0 & \text{otherwise}
\end{cases}
\]
3. 从 \( P_{\text{top-k}}(x) \) 中随机采样一个词。

---

### 3. **Top-k 采样的实现**

以下是一个简单的 Python 实现示例：

```python
import torch
import torch.nn.functional as F

def top_k_sampling(logits, k):
    # logits: 模型输出的未归一化概率分布，形状为 (vocab_size,)
    # k: 选择前 k 个词
    probs = F.softmax(logits, dim=-1)  # 将 logits 转换为概率分布
    top_k_probs, top_k_indices = torch.topk(probs, k)  # 选择前 k 个词的概率和索引
    top_k_probs = top_k_probs / top_k_probs.sum()  # 重新归一化
    sampled_index = torch.multinomial(top_k_probs, num_samples=1)  # 从 top-k 中采样
    return top_k_indices[sampled_index]  # 返回采样的词索引

```

### 示例：

```python
# 假设模型输出的 logits
logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# 使用 Top-k 采样（k=3）
sampled_index = top_k_sampling(logits, k=3)
print("Sampled index:", sampled_index.item())

```

---

### 4. **Top-k 采样的优点**

1. **减少低概率词的影响**：通过限制采样范围，避免选择概率极低的词，从而生成更连贯的文本。
2. **平衡多样性和质量**：相比于贪心搜索，Top-k 采样增加了文本的多样性；相比于随机采样，Top-k 采样提高了文本的质量。
3. **简单易实现**：只需一个参数 \( k \)，计算开销小。

---

### 5. **Top-k 采样的缺点**

1. **固定 \( k \) 的限制**：
    - 如果 \( k \) 设置过小，可能会导致生成的文本过于保守，缺乏多样性。
    - 如果 \( k \) 设置过大，可能会引入低概率词，影响文本质量。
2. **动态性不足**：Top-k 采样对每个时间步都使用固定的 \( k \)，无法根据上下文动态调整采样范围。

---

### 6. **Top-k 采样的改进：Top-p（Nucleus）采样**

为了克服 Top-k 采样的缺点，**Top-p 采样（也称为 Nucleus 采样）** 被提出。Top-p 采样不是固定选择前 \( k \) 个词，而是选择一个最小的词集合，使得这些词的概率之和大于等于 \( p \)（例如 \( p = 0.9 \)）。这样可以动态调整采样范围，适应不同的上下文。

---

### 7. **Top-k 采样的应用场景**

Top-k 采样广泛应用于以下任务：

- **文本生成**：如故事生成、对话生成、代码生成等。
- **机器翻译**：生成多样化的翻译结果。
- **语音识别**：生成多样化的转录结果。

---

### 8. **总结**

- **Top-k 采样** 是一种简单而有效的文本生成采样策略，通过限制采样范围来提高生成文本的质量和多样性。
- 它通过选择概率最高的前 \( k \) 个词，并对这些词重新归一化后进行采样。
- Top-k 采样的改进版本是 **Top-p 采样**，后者可以动态调整采样范围，适应不同的上下文。

如果你有更多问题，欢迎继续讨论！

## 加载模型Loading pretrained weights from OpenAI

```python

def download_and_load_gpt2(model_size, models_dir):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params

# Define model configurations in a dictionary for compactness
model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# Copy the base configuration and update with specific model settings
model_name = "gpt2-small (124M)"  # Example model name
NEW_CONFIG = GPT_CONFIG_124M.copy()
NEW_CONFIG.update(model_configs[model_name])
NEW_CONFIG.update({"context_length": 1024, "qkv_bias": True})

gpt = GPTModel(NEW_CONFIG)
gpt.eval();

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

import numpy as np

def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
    
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"])

        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale, 
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift, 
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale, 
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift, 
            params["blocks"][b]["ln_2"]["b"])

    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])
    
    
load_weights_into_gpt(gpt, params)
gpt.to(device)

torch.manual_seed(123)

token_ids = generate(
    model=gpt,
    idx=text_to_token_ids("Every effort moves you", tokenizer).to(device),
    max_new_tokens=25,
    context_size=NEW_CONFIG["context_length"],
    top_k=50,
    temperature=1.5
)

print("Output text:\n", token_ids_to_text(token_ids, tokenizer))
```

```
Output text:
 Every effort moves you toward finding an ideal new way to practice something!

What makes us want to be on top of that?
```

## 总结

LLMs的文本生成机制（贪婪解码、概率采样和温度缩放）、训练过程（损失函数和优化器）以及预训练的挑战和替代方案（使用公开的预训练权重）。这些技术和方法共同支撑了LLMs的高效训练和文本生成能力。

### 1. **文本生成过程**

- LLMs生成文本时，每次输出一个**标记（token）**。
- 默认情况下，模型通过将输出转换为概率分数，并选择概率最高的标记（称为**贪婪解码，greedy decoding**）来生成下一个标记。
- 为了提高生成文本的多样性和连贯性，可以使用**概率采样（probabilistic sampling）和温度缩放（temperature scaling）**。

---

### 2. **训练与验证**

- 训练和验证集的**损失值（loss）**用于评估LLM在训练过程中生成的文本质量。
- 训练LLM的目标是通过调整模型权重来最小化训练损失。
- 训练过程使用标准的深度学习流程，包括**交叉熵损失函数（cross entropy loss）和AdamW优化器**。

---

### 3. **预训练**

- 预训练LLM需要在一个大规模文本语料库上进行，这是一个**耗时且资源密集**的过程。
- 为了避免从头开始预训练，可以使用公开的预训练权重（如OpenAI提供的权重）作为替代方案。

---

# 总结
