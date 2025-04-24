# VidQuery

![GitHub](https://img.shields.io/github/license/yaseenmd/vidquery)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)

VidQuery is an AI-powered question answering system that helps users extract and understand information from YouTube videos using their transcripts. With VidQuery, you can ask specific questions about video content and receive precise answers without having to watch the entire video.

## Features

- **YouTube Transcript Processing**: Automatically extracts and processes captions from YouTube videos
- **Smart Q&A System**: Ask questions about video content and get AI-generated answers
- **Multi-language Support**: Works with videos in multiple languages through automatic translation
- **Memory-efficient Processing**: Optimized to handle long videos with minimal resource usage
- **Context-aware Responses**: Provides relevant answers with timestamp references when available
- **User-friendly Interface**: Simple Streamlit interface for easy interaction

## Demo

![VidQuery Demo](https://85015-01jsh1akav89yrrkh53gsn0t0c.cloudspaces.litng.ai)

## How It Works

1. Enter a YouTube URL
2. VidQuery extracts and processes the video transcript
3. Ask questions about the video content
4. Get accurate answers based specifically on information in the video

## Installation

### Prerequisites

- Python 3.8+
- A Hugging Face account and API token (for model access)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/vidquery.git
cd vidquery

# Install dependencies
pip install -r requirements.txt

# Create .env file with your Hugging Face token
echo "HF_TOKEN=your_huggingface_token_here" > .env

# Run the application
streamlit run app.py
```

### Environment Variables

Create a `.env` file in the project root with:

```
MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
HF_TOKEN=your_huggingface_token_here
```

## Technical Details

VidQuery uses a powerful pipeline to process videos and answer questions:

1. **Transcript Extraction**: Fetches captions using YouTube's transcript API
2. **Language Processing**: Detects language and translates non-English content
3. **Text Chunking & Embedding**: Breaks down transcript into manageable segments and converts to vector embeddings
4. **Semantic Search**: Identifies the most relevant parts of the transcript for each question
5. **Context-based Answer Generation**: Uses a language model to generate coherent answers from relevant context

## System Requirements

- **CPU**: Any modern multi-core processor
- **RAM**: Minimum 8GB, 16GB recommended
- **GPU** (optional): NVIDIA GPU with 4GB+ VRAM for faster processing
- **Storage**: 1GB free disk space for model caching

## Project Structure

```
vidquery/
├── app.py                  # Main Streamlit application
├── requirements.txt        # Required Python packages
└── utils/
    ├── __init__.py         # Package initialization
    ├── config.py           # Configuration settings
    ├── embedding.py        # Text embedding functionality
    ├── model.py            # Language model initialization
    ├── retriever.py        # Q&A chain implementation
    └── transcribe.py       # YouTube transcript extraction
```

## Dependencies

```
# Core
streamlit==1.32.0
langchain==0.1.9
langchain-community==0.0.24
langchain-core==0.1.27

# YouTube and transcript handling
youtube-transcript-api==0.6.1

# Language detection and translation
langdetect==1.0.9
deep-translator==1.11.4

# Embeddings and vector store
faiss-cpu==1.7.4
sentence-transformers==2.3.1

# LLM and tokenizer
transformers==4.38.2
torch==2.1.2
accelerate==0.27.2

# Utilities
uuid==1.30
python-dotenv==1.0.1
```

## Limitations

- Requires videos to have captions/transcripts available
- Answer quality depends on transcript accuracy and completeness
- Processing very long videos (>2 hours) may require significant resources
- Relies on third-party services for translation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Hugging Face](https://huggingface.co/) for model hosting and access
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api) for transcript extraction
- [LangChain](https://www.langchain.com/) for the LLM implementation framework
- [Streamlit](https://streamlit.io/) for the web interface
