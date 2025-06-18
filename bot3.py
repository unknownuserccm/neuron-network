import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import discord
from discord.ext import commands
import os
import json
from tqdm import tqdm

# ===== Device Setup =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Using device: {device}")

# ===== Enhanced Dataset with Auto-Fixing =====
class SmartChatDataset(Dataset):
    def __init__(self, filepath, existing_vocab=None):
        self.pairs = []
        
        # Initialize vocabulary - either from existing or start fresh
        if existing_vocab:
            self.word2idx = existing_vocab['word2idx'].copy()
            self.idx2word = existing_vocab['idx2word'].copy()
            # Convert string keys to int for idx2word (auto-fix corruption)
            self.idx2word = {int(k): v for k, v in self.idx2word.items()}
            self.index = max(self.word2idx.values()) + 1
            print(f"📚 Loaded existing vocabulary: {len(self.word2idx)} words")
        else:
            self.word2idx = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
            self.idx2word = {0: '<pad>', 1: '<sos>', 2: '<eos>', 3: '<unk>'}
            self.index = 4
            print("🆕 Creating new vocabulary")

        # Load and process data
        self._load_data(filepath)
        print(f"✅ Dataset ready: {len(self.pairs)} pairs, {len(self.word2idx)} vocabulary")

    def _load_data(self, filepath):
        """Load data with robust error handling"""
        if not os.path.exists(filepath):
            print(f"❌ File {filepath} not found!")
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        print("🔄 Loading dataset...")
        valid_count = 0
        
        for line in tqdm(lines, desc="Processing data"):
            line = line.strip()
            if not line or '\t' not in line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 2:
                inp, out = parts[0].strip(), parts[1].strip()
                if inp and out:  # Skip empty
                    inp_tokens = self._tokenize(inp)
                    out_tokens = self._tokenize(out, add_eos=True)
                    if inp_tokens and out_tokens:  # Valid tokens
                        self.pairs.append((inp_tokens, out_tokens))
                        valid_count += 1

        print(f"📊 Processed {valid_count} valid conversation pairs")

    def _tokenize(self, sentence, add_eos=False):
        """Smart tokenization with auto-vocabulary building"""
        # Clean and split
        words = sentence.lower().strip().split()
        if add_eos:
            words.append('<eos>')
        
        indices = []
        for word in words:
            # Clean word (remove punctuation)
            clean_word = ''.join(c for c in word if c.isalnum()).strip()
            if not clean_word:
                continue
                
            # Add to vocabulary if new
            if clean_word not in self.word2idx:
                self.word2idx[clean_word] = self.index
                self.idx2word[self.index] = clean_word
                self.index += 1
            
            indices.append(self.word2idx[clean_word])
        
        return indices if indices else [self.word2idx['<unk>']]

    def get_vocab(self):
        """Return vocabulary dictionaries"""
        return {'word2idx': self.word2idx, 'idx2word': self.idx2word}

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return torch.tensor(self.pairs[idx][0]), torch.tensor(self.pairs[idx][1])

# ===== Collate function =====
def collate_fn(batch):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=0)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=0)
    return src_padded.to(device), tgt_padded.to(device)

# ===== Auto-Resizing Model =====
class AutoFixChatBot(nn.Module):
    def __init__(self, vocab_size, embed_size=128, hidden_size=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.encoder = nn.GRU(embed_size, hidden_size, batch_first=True, bidirectional=True)
        self.decoder = nn.GRU(embed_size, hidden_size * 2, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)
        
        print(f"🤖 Model created: vocab={vocab_size}, embed={embed_size}, hidden={hidden_size}")

    def forward(self, src, tgt_input):
        # Encode
        embedded_src = self.embedding(src)
        _, hidden = self.encoder(embedded_src)
        
        # Handle bidirectional hidden state
        if hidden.size(0) == 2:  # bidirectional
            hidden = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)

        # Decode
        embedded_tgt = self.embedding(tgt_input)
        output, _ = self.decoder(embedded_tgt, hidden)
        output = self.fc(output)
        return output

    def auto_resize_vocab(self, new_vocab_size):
        """Automatically resize vocabulary - handles both expansion and shrinking"""
        if new_vocab_size == self.vocab_size:
            print("✅ Vocabulary size unchanged")
            return

        print(f"🔧 Auto-resizing vocabulary: {self.vocab_size} → {new_vocab_size}")
        
        # Save current weights
        old_embed_weight = self.embedding.weight.data.clone()
        old_fc_weight = self.fc.weight.data.clone()
        old_fc_bias = self.fc.bias.data.clone()

        # Create new layers
        new_embedding = nn.Embedding(new_vocab_size, self.embed_size, padding_idx=0)
        new_fc = nn.Linear(self.hidden_size * 2, new_vocab_size)

        # Copy weights safely
        copy_size = min(self.vocab_size, new_vocab_size)
        
        # Copy existing weights
        new_embedding.weight.data[:copy_size] = old_embed_weight[:copy_size]
        new_fc.weight.data[:copy_size] = old_fc_weight[:copy_size]
        new_fc.bias.data[:copy_size] = old_fc_bias[:copy_size]

        # Initialize new weights if expanding
        if new_vocab_size > self.vocab_size:
            with torch.no_grad():
                # Initialize new embeddings
                nn.init.normal_(new_embedding.weight.data[self.vocab_size:], 0, 0.1)
                # Initialize new output weights
                nn.init.normal_(new_fc.weight.data[self.vocab_size:], 0, 0.1)
                nn.init.zeros_(new_fc.bias.data[self.vocab_size:])

        # Replace layers
        self.embedding = new_embedding
        self.fc = new_fc
        self.vocab_size = new_vocab_size
        
        print(f"✅ Vocabulary resized successfully to {new_vocab_size}")

# ===== Training with Auto-Fixing =====
def train_model(dataset, model, num_epochs=30):
    """Train model with automatic fixing"""
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)

    print(f"🚀 Training for {num_epochs} epochs...")
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for src, tgt in progress:
            # Prepare target input (teacher forcing)
            tgt_input = torch.cat([
                torch.full((tgt.size(0), 1), dataset.word2idx['<sos>'], device=device),
                tgt[:, :-1]
            ], dim=1)

            # Forward pass
            pred = model(src, tgt_input)
            loss = criterion(pred.view(-1, pred.size(-1)), tgt.view(-1))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"✅ Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")

    return avg_loss

# ===== Smart Inference =====
def smart_reply(message, model, word2idx, idx2word, max_len=25):
    """Generate response with better error handling"""
    model.eval()
    
    # Clean and tokenize input
    words = message.lower().strip().split()
    clean_words = []
    for word in words:
        clean_word = ''.join(c for c in word if c.isalnum()).strip()
        if clean_word:
            clean_words.append(clean_word)
    
    if not clean_words:
        return "I didn't understand that."
    
    # Convert to indices
    input_indices = []
    unknown_count = 0
    for word in clean_words:
        if word in word2idx:
            input_indices.append(word2idx[word])
        else:
            input_indices.append(word2idx['<unk>'])
            unknown_count += 1
    
    # If too many unknown words, give generic response
    if unknown_count > len(clean_words) * 0.7:
        return "I'm not familiar with those words yet. Can you try simpler words?"

    input_tensor = torch.tensor(input_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        # Encode
        embedded = model.embedding(input_tensor)
        _, hidden = model.encoder(embedded)
        
        # Handle bidirectional hidden state
        if hidden.size(0) == 2:
            hidden = torch.cat([hidden[0], hidden[1]], dim=1).unsqueeze(0)

        # Decode
        next_input = torch.tensor([[word2idx['<sos>']]]).to(device)
        response = []

        for _ in range(max_len):
            out, hidden = model.decoder(model.embedding(next_input), hidden)
            pred = model.fc(out[:, -1])
            next_word = pred.argmax(dim=-1).item()

            # Stop at end token
            if next_word == word2idx.get('<eos>', 2):
                break

            # Get word (with safety check)
            word = idx2word.get(next_word, '<unk>')
            if word not in ['<pad>', '<sos>', '<unk>']:
                response.append(word)

            next_input = torch.tensor([[next_word]]).to(device)

    if not response:
        return "I'm not sure how to respond to that yet."
    
    return ' '.join(response)

# ===== Save/Load with Auto-Fixing =====
def save_model(model, vocab, filepath="smart_model.pth"):
    """Save model with vocabulary"""
    print("💾 Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab': vocab,
        'vocab_size': model.vocab_size,
        'embed_size': model.embed_size,
        'hidden_size': model.hidden_size
    }, filepath)
    print(f"✅ Model saved to {filepath}")

def load_model(filepath="smart_model.pth"):
    """Load model with auto-fixing"""
    if not os.path.exists(filepath):
        return None, None
    
    print("📦 Loading model...")
    data = torch.load(filepath, map_location=device)
    
    # Auto-fix vocabulary if corrupted
    vocab = data['vocab']
    if 'idx2word' in vocab:
        # Fix string keys to int keys
        vocab['idx2word'] = {int(k): v for k, v in vocab['idx2word'].items()}
    
    # Create model with saved parameters
    model = AutoFixChatBot(
        data['vocab_size'],
        data.get('embed_size', 128),
        data.get('hidden_size', 256)
    ).to(device)
    
    model.load_state_dict(data['model_state_dict'])
    print(f"✅ Model loaded successfully")
    return model, vocab

# ===== Discord Bot =====
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

# Global variables
model = None
vocab = None

@bot.event
async def on_ready():
    print(f"🤖 Bot logged in as {bot.user}")

@bot.event
async def on_message(message):
    global model, vocab
    if message.author.bot:
        return

    try:
        if model is None or vocab is None:
            await message.channel.send("🔄 Bot is still loading, please wait...")
            return

        response = smart_reply(message.content, model, vocab['word2idx'], vocab['idx2word'])
        
        if not response.strip():
            response = "Sorry, I don't know how to respond yet. Try teaching me more!"
        
        await message.channel.send(response)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        await message.channel.send("Oops! Something went wrong. Please try again.")

    await bot.process_commands(message)

# ===== Main with Auto-Fixing =====
if __name__ == "__main__":
    # Try to load existing model
    model, vocab = load_model()
    
    if model and vocab:
        choice = input(
            "📁 Existing model found! Choose option:\n"
            "1 - Continue training (add new data and train more)\n"
            "2 - Load model without training\n"
            "3 - Train completely new model\n"
            "Enter 1, 2, or 3: "
        ).strip()
        
        if choice == "1":
            print("🔄 Loading dataset and continuing training...")
            # Load dataset with existing vocabulary
            dataset = SmartChatDataset("data.txt", vocab)
            
            # Auto-resize model if vocabulary changed
            model.auto_resize_vocab(len(dataset.word2idx))
            model.to(device)
            
            # Update vocabulary reference
            vocab = dataset.get_vocab()
            
            # Continue training
            train_model(dataset, model, num_epochs=10)
            save_model(model, vocab)
            
        elif choice == "2":
            print("📚 Loading model without additional training...")
            model.eval()
            
        elif choice == "3":
            print("🆕 Training completely new model...")
            dataset = SmartChatDataset("data.txt")
            model = AutoFixChatBot(len(dataset.word2idx)).to(device)
            vocab = dataset.get_vocab()
            
            train_model(dataset, model, num_epochs=30)
            save_model(model, vocab)
        else:
            print("📚 Invalid choice, loading existing model...")
            model.eval()
    else:
        print("🆕 No existing model found. Training new model...")
        dataset = SmartChatDataset("data.txt")
        model = AutoFixChatBot(len(dataset.word2idx)).to(device)
        vocab = dataset.get_vocab()
        
        train_model(dataset, model, num_epochs=30)
        save_model(model, vocab)

    # Get Discord token
    TOKEN = ""
    bot.run(TOKEN)