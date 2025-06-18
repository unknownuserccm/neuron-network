# 🤖 Smart Discord Chatbot

A sophisticated Discord chatbot powered by a custom neural network with auto-fixing capabilities. This bot learns from conversation data and can engage in natural conversations with users.

## ✨ Features

- **🧠 Neural Network Powered**: Uses a custom GRU-based encoder-decoder architecture
- **🔧 Auto-Fixing**: Automatically handles vocabulary corruption and model resizing
- **📚 Smart Dataset Loading**: Robust data processing with error handling
- **🎯 Adaptive Training**: Supports continuing training with new data
- **💬 Natural Conversations**: Generates contextually appropriate responses
- **📊 Progress Tracking**: Real-time training progress with tqdm
- **🛡️ Error Handling**: Comprehensive error handling and recovery

## 🚀 Quick Start

### Prerequisites

```bash
pip install torch discord.py tqdm
```

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd discord-chatbot
   ```

2. **Prepare your training data**
   - Create a `data.txt` file with conversation pairs
   - Format: `input_message\tresponse_message` (tab-separated)
   - Example:
     ```
     hello	hi there how are you
     how are you	im doing great thanks
     what is your name	i am a chatbot
     ```

3. **Get Discord Bot Token**
   - Go to [Discord Developer Portal](https://discord.com/developers/applications)
   - Create a new application and bot
   - Copy the bot token
   - Add the token to the `TOKEN` variable in the script

4. **Run the bot**
   ```bash
   python chatbot.py
   ```

## 📁 Project Structure

```
discord-chatbot/
├── chatbot.py          # Main bot script
├── data.txt            # Training data (tab-separated)
├── smart_model.pth     # Saved model (generated after training)
└── README.md           # This file
```

## 🛠️ How It Works

### Architecture

- **Encoder**: Bidirectional GRU that processes input messages
- **Decoder**: Unidirectional GRU that generates responses
- **Embedding Layer**: Converts words to dense vectors
- **Output Layer**: Maps hidden states to vocabulary probabilities

### Smart Features

1. **Auto-Vocabulary Building**: Automatically builds vocabulary from training data
2. **Model Resizing**: Dynamically resizes the model when vocabulary changes
3. **Corruption Recovery**: Automatically fixes corrupted vocabulary files
4. **Incremental Training**: Add new data and continue training existing models

## 🎮 Usage

### Training Options

When you run the script, you'll see these options:

1. **Continue Training**: Load existing model and train with new data
2. **Load Model**: Use existing model without additional training
3. **New Model**: Train a completely new model from scratch

### Discord Commands

The bot responds to all messages (except from other bots) and generates contextually appropriate replies based on its training.

## ⚙️ Configuration

### Model Parameters

```python
# Adjust these in the AutoFixChatBot class
embed_size = 128      # Embedding dimension
hidden_size = 256     # Hidden layer size
```

### Training Parameters

```python
# Modify in train_model function
batch_size = 32       # Batch size for training
num_epochs = 30       # Number of training epochs
learning_rate = 0.001 # Learning rate
```

## 📊 Training Data Format

Your `data.txt` should contain tab-separated conversation pairs:

```
hello	hi there how can i help you
goodbye	see you later have a great day
what time is it	i dont have access to the current time
tell me a joke	why did the chicken cross the road to get to the other side
```

**Tips for better training data:**
- Use diverse conversation examples
- Include common greetings and responses
- Add domain-specific conversations if needed
- Ensure proper grammar and spelling
- Include varied response lengths

## 🔧 Advanced Features

### Auto-Fixing Capabilities

- **Vocabulary Corruption**: Automatically fixes corrupted word-to-index mappings
- **Model Compatibility**: Handles model loading across different vocabulary sizes
- **Graceful Degradation**: Continues working even with incomplete data

### Smart Response Generation

- **Unknown Word Handling**: Gracefully handles words not in vocabulary
- **Context Awareness**: Uses encoder-decoder architecture for better context
- **Response Filtering**: Filters out special tokens and empty responses

## 🐛 Troubleshooting

### Common Issues

1. **"CUDA out of memory"**
   - Reduce batch size in `train_model` function
   - Use CPU by setting `device = torch.device("cpu")`

2. **Bot not responding**
   - Check Discord token is correct
   - Ensure bot has message permissions in the server
   - Verify the model loaded successfully

3. **Poor response quality**
   - Add more diverse training data
   - Increase training epochs
   - Adjust model parameters (embed_size, hidden_size)

### Debug Mode

Add this to enable detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Tips

- **GPU Usage**: Automatically uses CUDA if available
- **Memory Management**: Implements gradient clipping and proper batch processing
- **Learning Rate Scheduling**: Uses ReduceLROnPlateau for better convergence

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with PyTorch for neural network implementation
- Uses Discord.py for bot functionality
- Inspired by sequence-to-sequence models for natural language processing

